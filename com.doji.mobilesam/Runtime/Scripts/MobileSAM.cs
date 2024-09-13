using System;
using Unity.Sentis;
using UnityEngine;
using static Doji.AI.Segmentation.MobileSAM.Transforms;

namespace Doji.AI.Segmentation {

    public struct DecoderOutput {
        public Tensor<float> LowResMasks { get; private set; }
        public Tensor<float> IoUPredictions { get; private set; }
        public Tensor<float> Masks { get; private set; }

        public DecoderOutput(Tensor lowResMasks, Tensor iouPredictions, Tensor masks) {
            LowResMasks = lowResMasks as Tensor<float>;
            IoUPredictions = iouPredictions as Tensor<float>;
            Masks = masks as Tensor<float>;
        }
    }

    /// <summary>
    /// Predictor using MobileSAM models.
    /// Call <see cref="SetImage(Texture)"/> to specify the image you want to generate masks for.
    /// After setting an image you can call <see cref="Predict(float[], float[], Rect?, Texture)"/>
    /// to get the masks. The results will be stored in the <see cref="Result"/> RenderTexture.
    /// </summary>
    public partial class MobileSAM : IDisposable {

        public Model Encoder {
            get {
                if (_encoderAsset == null) {
                    _encoderAsset = Resources.Load<ModelAsset>("ONNX/mobilesam.encoder");
                }
                if (_encoderAsset == null) {
                    Debug.LogError("MobileSAM encoder ONNX model not found.");
                }
                return _encoderModel ??= ModelLoader.Load(_encoderAsset);
            }
        }
        private Model _encoderModel;
        private static ModelAsset _encoderAsset;

        public Model Decoder {
            get {
                if (_decoderAsset == null) {
                    _decoderAsset = Resources.Load<ModelAsset>("ONNX/mobilesam.decoder");
                }
                if (_decoderAsset == null) {
                    Debug.LogError("MobileSAM decoder ONNX model not found.");
                }
                return _decoderModel ??= ModelLoader.Load(_decoderAsset);
            }
        }
        private static Model _decoderModel;
        private ModelAsset _decoderAsset;

        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        public BackendType Backend {
            get => _backend;
            set {
                if (_backend != value) {
                    DisposeModels();
                    _backend = value;
                    InitializeNetwork();
                }
            }
        }
        private BackendType _backend = BackendType.CPU;

        /// <summary>
        /// Which format the predicted masks are returned as.
        /// </summary>
        public RenderTextureFormat OutputFormat { get; set; } = RenderTextureFormat.ARGB32;

        private Worker _encoder;
        private Worker _decoder;
        private readonly Tensor[] _inputTensors = new Tensor[6];

        private bool _isImageSet;
        private (int height, int width) _origSize;
        private Vector2Int _inputSize;
        private Tensor _features;

        public RenderTexture Result { get; private set; }

        /// <summary>
        /// Initializes a new instance of a mask predictor.
        /// </summary>
        public MobileSAM() {
            InitializeNetwork();
        }

        private void InitializeNetwork() {
            if (Encoder == null) {
                return;
            }
            if (Decoder == null) {
                return;
            }

            _encoderModel = AddPreprocessing(_encoderModel);

            _encoder = new Worker(Encoder, Backend);
            _decoder = new Worker(Decoder, Backend);
        }

        /// <summary>
        /// Specifies the image to be used for mask generation using the
        /// <see cref="Predict(float[], float[], Rect?, Texture)"/> method.
        /// </summary>
        /// <remarks>
        /// This encodes the input image and stores the calculated image embeddings.
        /// The image only needs to be encoded once. Afterwards, you can query
        /// as many masks as you want.
        /// </remarks>
        public void SetImage(Texture image) {
            var origSize = (image.height, image.width);
            // Transform the image to the form expected by the model
            using Tensor inputImageTensor = ApplyImage(image);
            SetImage(inputImageTensor, origSize);
            if (Result != null) Result.Release();   
            Result = new RenderTexture(image.width, image.height, 0, OutputFormat);
        }

        public void SetImage(Tensor transformedImage, (int height, int width) origSize) {
            Debug.Assert(Math.Max(transformedImage.shape[0], transformedImage.shape[1]) == IMG_SIZE, "Image must have a long side of {IMG_SIZE}}.");
            ResetImage();
            _origSize = origSize;
            _inputSize = new Vector2Int(transformedImage.shape[-2], transformedImage.shape[-1]);
            // preprocessing/normalization is already baked into the model
            _encoder.Schedule(transformedImage);
            _features = _encoder.PeekOutput();
            _isImageSet = true;
        }

        /// <summary>
        /// Resets the currently set image.
        /// </summary>
        private void ResetImage() {
            _isImageSet = false;
            _features = null;
            _origSize = default;
            _inputSize = default;
        }

        /// <summary>
        /// Predict masks for the given input prompts, using the currently set image.
        /// </summary>
        /// <param name="pointCoords">A float array of point prompts to the model.
        /// Each point is given in pixel coordinates.</param>
        /// <param name="pointLabels">A length N array of labels for the point prompts.
        /// 1 indicates a foreground point and 0 indicates a background point.</param>
        public void Predict(float[] pointCoords, float[] pointLabels) {
            if (!_isImageSet) {
                throw new InvalidOperationException("An image must be set with .SetImage(...) before mask prediction.");
            }
            if (pointCoords == null) {
                throw new ArgumentNullException("pointCoords can not be null.", nameof(pointCoords));
            }
            if (pointLabels == null) {
                throw new ArgumentNullException("pointLabels can not be null.", nameof(pointLabels));
            }

            int numPoints = pointCoords?.Length / 2 ?? 0;
            int numLabels = pointLabels?.Length ?? 0;
            if (numPoints != numLabels) {
                throw new ArgumentException("number of point labels does not match the number of points.");
            }
            pointCoords = ApplyCoords(pointCoords, _origSize);
            using Tensor<float> point_coords = new Tensor<float>(new TensorShape(1, numPoints, 2), pointCoords);
            using Tensor<float> point_labels = new Tensor<float>(new TensorShape(1, numPoints), pointLabels);
            using Tensor<float> mask_input = new Tensor<float>(new TensorShape(1, 1, 256, 256));
            using Tensor<float> has_mask_input = new Tensor<float>(new TensorShape(1), new float[] { 0f });
            using Tensor<float> orig_im_size = new Tensor<float>(new TensorShape(2), new float[] { _origSize.height, _origSize.width });

            var result = Predict(point_coords, point_labels, mask_input, has_mask_input, orig_im_size);
            TextureConverter.RenderToTexture(result.Masks, Result, new TextureTransform().SetBroadcastChannels(true));
        }

        private DecoderOutput Predict(
            Tensor pointCoords,
            Tensor pointLabels,
            Tensor maskInput,
            Tensor hasMaskInput,
            Tensor origImSize)
        {
            _inputTensors[0] = _features;       // image_embeddings
            _inputTensors[1] = pointCoords;     // point_coords    
            _inputTensors[2] = pointLabels;     // point_labels    
            _inputTensors[3] = maskInput;       // mask_input      
            _inputTensors[4] = hasMaskInput;    // has_mask_input  
            _inputTensors[5] = origImSize;      // orig_im_size    

            _decoder.Schedule(_inputTensors);
            var lowResMasks = _decoder.PeekOutput("low_res_masks");
            var iouPredictions = _decoder.PeekOutput("iou_predictions");
            var masks = _decoder.PeekOutput("masks");

            return new DecoderOutput(lowResMasks, iouPredictions, masks);
        }

        /// <summary>
        /// Normalize pixel values and pad to a square input.
        /// </summary>
        private Model AddPreprocessing(Model model) {
            var graph = new FunctionalGraph();
            var inputs = graph.AddInputs(model);
            // [0, 1] -> [0, 255]
            inputs[0] *= 255f;
            FunctionalTensor[] outputs = Functional.Forward(model, inputs);
            return graph.Compile(outputs);
        }

        public void Dispose() {
            DisposeModels();
            Result.Release();
        }

        private void DisposeModels() {
            _encoder?.Dispose();
            _decoder?.Dispose();
        }
    }
}