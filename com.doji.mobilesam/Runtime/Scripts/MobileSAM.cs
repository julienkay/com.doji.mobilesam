using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Segmentation {

    public class MobileSAM : IDisposable {

        public static Model Encoder {
            get {
                if (_encoderAsset == null) {
                    _encoderAsset = Resources.Load<ModelAsset>("ONNX/mobilesam.encoder");
                }
                if (_encoderAsset == null) {
                    Debug.LogError("MobileSAM encoder ONNX model not found.");
                } else if (_encoderModel == null) {
                    _encoderModel = ModelLoader.Load(_encoderAsset);
                }
                return _encoderModel;
            }
        }
        private static Model _encoderModel;
        private static ModelAsset _encoderAsset;

        public static Model Decoder {
            get {
                if (_decoderAsset == null) {
                    _decoderAsset = Resources.Load<ModelAsset>("ONNX/mobilesam.decoder");
                }
                if (_decoderAsset == null) {
                    Debug.LogError("MobileSAM decoder ONNX model not found.");
                } else if (_decoderModel == null) {
                    _decoderModel = ModelLoader.Load(_decoderAsset);
                }
                return _decoderModel;
            }
        }
        private static Model _decoderModel;
        private static ModelAsset _decoderAsset;


        /// <summary>
        /// Which <see cref="BackendType"/> to run the model with.
        /// </summary>
        public BackendType Backend {
            get => _backend;
            set {
                if (_backend != value) {
                    Dispose();
                    _backend = value;
                    InitializeNetwork();
                }
            }
        }
        private BackendType _backend = BackendType.CPU;

        /// <summary>
        /// Whether to normalize the estimated depth.
        /// </summary>
        /// <remarks>
        /// MiDaS predicts depth values as inverse relative depth.
        /// (small values for far away objects, large values for near objects)
        /// If NormalizeDepth is enabled, these values are mapped to the (0, 1) range,
        /// which is mostly useful for visualization.
        /// </remarks>
        public bool NormalizeDepth { get; set; } = true;

        private Worker _encoder;
        private Worker _decoder;
        private Tensor[] _inputTensors = new Tensor[6];

        /// <summary>
        /// the (possibly resized) input texture;
        /// </summary>
        public RenderTexture _resizedInput;

        /// <summary>
        /// Initializes a new instance of MiDaS.
        /// </summary>
        /// <param name="model">the reference to a MiDaS ONNX model</param>
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

            var graph = new FunctionalGraph();
            var inputs = graph.AddInputs(Encoder);
            inputs[0] *= 255f;
            FunctionalTensor[] outputs = Functional.Forward(Encoder, inputs);
            _encoderModel = graph.Compile(outputs);

            _encoder = new Worker(Encoder, Backend);
            _decoder = new Worker(Decoder, Backend);
            _resizedInput = new RenderTexture(1024, 1024, 0);

            Debug.Log(_encoderModel.inputs.Count);
            Debug.Log(_encoderModel.inputs[0].name);
            Debug.Log(_encoderModel.inputs[0].shape);
        }

        public void PredictMasks(Texture input) {
            Tensor embeddings = EncodeImage(input);
            DecodeMasks(input, embeddings);
        }

        /// <summary>
        /// Encodes the input image and outputs an image embedding as a Tensor.
        /// </summary>
        private Tensor EncodeImage(Texture input) {
            using Tensor inputTensor = TextureConverter.ToTensor(input, input.width, input.height, 3);
            inputTensor.Reshape(new TensorShape(input.height, input.width, 3));
            _encoder.Schedule(inputTensor);
            Tensor embeddings = _encoder.PeekOutput();
            return embeddings;
        }

        /// <summary>
        /// Predict masks for the given input prompts, using the given <paramref name="imageEmbeddings"/>s.
        /// </summary>
        /// <param name="imageEmbeddings"></param>
        /// <param name="pointCoords">A Vector2 array of point prompts to the model.
        /// Each point is in (X, Y) in pixels.</param>
        /// <param name="pointLabels">A length N array of labels for the point prompts.
        /// 1 indicates a foreground point and 0 indicates a background point.</param>
        /// <param name="box">a box prompt to the model.</param>
        /// <param name="maskInput">A low resolution mask input to the model, typically
        /// coming from a previous prediction iteration.</param>
        /// <param name="multimask_output">If true, the model will return three masks.
        /// For ambiguous input prompts(such as a single click), this will often
        /// produce better masks than a single prediction.If only a single
        /// mask is needed, the model's predicted quality score can be used
        /// to select the best mask.For non-ambiguous prompts, such as multiple
        /// input prompts, multimask_output=False can give better results.</param>
        /// <param name="returnLogits">If true, returns un-thresholded masks logits
        /// instead of a binary mask.</param>
        private void DecodeMasks(
            Texture inputImage,
            Tensor imageEmbeddings,
            float[] pointCoords = null,
            float[] pointLabels = null,
            Rect? box = null,
            Texture maskInput = null,
            bool multimaskOutput = true,
            bool returnLogits = false
        ) {
            pointCoords = new float[] { inputImage.width / 2f, inputImage.height / 2f };
            pointLabels = new float[] { 1f };

            int numPoints = pointCoords?.Length / 2 ?? 0;
            int numLabels = pointLabels?.Length ?? 0;
            if (numPoints != numLabels) {
                throw new ArgumentException("number of point labels does not match the number of points.");
            }

            using Tensor<float> point_coords = new Tensor<float>(new TensorShape(1, numPoints, 2), pointCoords);
            using Tensor<float> point_labels = new Tensor<float>(new TensorShape(1, numPoints), pointLabels);
            using Tensor<float> mask_input = new Tensor<float>(new TensorShape(1, 1, 256, 256));
            using Tensor<float> has_mask_input = new Tensor<float>(new TensorShape(1), new float[] { 0f });
            using Tensor<float> orig_im_size = new Tensor<float>(new TensorShape(2), new float[] { inputImage.width, inputImage.height });

            _inputTensors[0] = imageEmbeddings; // image_embeddings
            _inputTensors[1] = point_coords;    // point_coords    
            _inputTensors[2] = point_labels;    // point_labels    
            _inputTensors[3] = mask_input;      // mask_input      
            _inputTensors[4] = has_mask_input;  // has_mask_input  
            _inputTensors[5] = orig_im_size;    // orig_im_size    

            _decoder.Schedule(_inputTensors);
            Tensor masks = _decoder.PeekOutput("masks");
            Debug.Log("Mask output shape " + masks.shape);
            int height = masks.shape[2];
            int width = masks.shape[3];

            Result = TextureConverter.ToTexture(masks as Tensor<float>, width, height);

        }
        
        public Texture Result { get; private set; }
        
        public void Dispose() {
            _encoder?.Dispose();
            _decoder?.Dispose();
        }
    }
}