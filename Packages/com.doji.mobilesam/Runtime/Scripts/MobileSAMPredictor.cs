using System;
using System.Collections.Generic;
using Unity.Sentis;
using Unity.Sentis.Layers;
using UnityEngine;

namespace MobileSAM {

    public class MobileSAMPredictor : IDisposable {

        public static Model Encoder {
            get {
                if (_encoderAsset == null) {
                    _encoderAsset = Resources.Load<ModelAsset>("ONNX/mobilesam.encoder");
                }
                if (_encoderAsset == null) {
                    Debug.LogError("MobileSAM encoder ONNX model not found.");
                } else {
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
                    _decoderAsset = Resources.Load<ModelAsset>("ONNX/sam_onnx_example");
                }
                if (_decoderAsset == null) {
                    Debug.LogError("MobileSAM decoder ONNX model not found.");
                } else {
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

        private IWorker _encoder;
        private IWorker _decoder;
        private ITensorAllocator _allocator;
        private Ops _ops;
        private Dictionary<string, Tensor> _inputTensors = new Dictionary<string, Tensor>();

        /// <summary>
        /// the (possibly resized) input texture;
        /// </summary>
        public RenderTexture _resizedInput;

        /// <summary>
        /// Initializes a new instance of MiDaS.
        /// </summary>
        /// <param name="model">the reference to a MiDaS ONNX model</param>
        public MobileSAMPredictor() {
            InitializeNetwork();
        }

        private void InitializeNetwork() {
            if (Encoder == null) {
                return;
            }
            if (Decoder == null) {
                return;
            }

            _encoder = WorkerFactory.CreateWorker(Backend, Encoder);
            _decoder = WorkerFactory.CreateWorker(Backend, Decoder);
            _allocator = new TensorCachingAllocator();
            _ops = WorkerFactory.CreateOps(Backend, _allocator);

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
            using (Tensor tensor = TextureConverter.ToTensor(input, input.width, input.height, 3)) {

                using (Tensor reshaped = tensor.ShallowReshape(new TensorShape(input.height, input.width, 3))) {
                    TensorFloat r = _ops.Mul(reshaped as TensorFloat, 255f);

                    /*using (Tensor reshaped = _ops.Reshape(tensor, new TensorShape(input.height, input.width, 3))) {
                        reshaped.MakeReadable();
                        Debug.Log(reshaped.shape);
                        _worker.Execute(reshaped);
                    }*/
                    /*var backend = _worker.GetBackend();
                    backend.NewTempTensorFloat(new TensorShape(input.height, input.width));
                    backend.Sq
                    Squeeze squeeze = new Squeeze("SqueezeInput", "input_image");
                    squeeze.Execute(tensor,)*/
                    _encoder.Execute(r);
                }
            }

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

            TensorFloat point_coords = new TensorFloat(new TensorShape(1, numPoints, 2), pointCoords);
            TensorFloat point_labels = new TensorFloat(new TensorShape(1, numPoints), pointLabels);
            TensorFloat mask_input = TensorFloat.Zeros(new TensorShape(1, 1, 256, 256));
            TensorFloat has_mask_input = new TensorFloat(new TensorShape(1), new float[] { 0f });
            TensorFloat orig_im_size = new TensorFloat(new TensorShape(2), new float[] { inputImage.width, inputImage.height });

            _inputTensors.Clear();
            _inputTensors["image_embeddings"] = imageEmbeddings;
            _inputTensors["point_coords"    ] = point_coords;
            _inputTensors["point_labels"    ] = point_labels;
            _inputTensors["mask_input"      ] = mask_input;
            _inputTensors["has_mask_input"  ] = has_mask_input;
            _inputTensors["orig_im_size"    ] = orig_im_size;

            _decoder.Execute(_inputTensors);
            Tensor masks = _decoder.PeekOutput("masks");
            Debug.Log("Mask output shape " + masks.shape);
            int height = masks.shape[2];
            int width = masks.shape[3];

            Result = TextureConverter.ToTexture(masks as TensorFloat, width, height);

        }
        public Texture Result { get; private set; }
        public void Dispose() {
            _encoder?.Dispose();
            _decoder?.Dispose();

            _allocator?.Dispose();
            _ops?.Dispose();
        }
    }
}