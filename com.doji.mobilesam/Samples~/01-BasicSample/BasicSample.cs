using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Doji.AI.Segmentation.Samples {

    public class MobileSAM_BasicSample : MonoBehaviour, IPointerDownHandler {

        private MobileSAM _mobileSAMPredictor;

        public Texture2D SampleImage;
        public RawImage SourceImage;
        public RawImage MaskImage;
        public RectTransform DebugMarker;
        public RenderTexture Result;

        public void Start () {
            _mobileSAMPredictor = new MobileSAM();
            _mobileSAMPredictor.SetImage(SourceImage.texture);
            Result = _mobileSAMPredictor.Result;
            MaskImage.texture = _mobileSAMPredictor.Result;
        }

        private void OnDestroy() {
            _mobileSAMPredictor?.Dispose();
        }

        public void PredictMask(Vector2 point) {
            var points = new float[] { point.x, point.y };
            var labels = new float[] { 1f /* foreground */ };
            _mobileSAMPredictor.Predict(points, labels);
            MaskImage.enabled = true;
        }
      
        public void OnPointerDown(PointerEventData eventData) {
            if (eventData.button == PointerEventData.InputButton.Left) {
                RectTransform rectTransform = SourceImage.GetComponent<RectTransform>();

                Vector2 localPoint;
                RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, eventData.position, eventData.pressEventCamera, out localPoint);

                Rect rect = rectTransform.rect;
                float normalizedX = (localPoint.x - rect.x) / rect.width;
                float normalizedY = (localPoint.y - rect.y) / rect.height;

                Vector2 uvPosition = new Vector2(normalizedX, 1.0f - normalizedY);
                Vector2 textureSpacePos = new Vector2(SourceImage.texture.width, SourceImage.texture.height) * uvPosition;

                // Ensure click was inside of the texture area
                if (uvPosition.x >= 0 && uvPosition.x <= 1 && uvPosition.y >= 0 && uvPosition.y <= 1) {
                    MoveDebugMarker(uvPosition);
                    PredictMask(textureSpacePos);
                }
            }
        }

        private void MoveDebugMarker(Vector2 uvPosition) {
            if (DebugMarker != null) {
                uvPosition.y = 1f - uvPosition.y;
                DebugMarker.anchorMin = uvPosition;
                DebugMarker.anchorMax = uvPosition;
                DebugMarker.gameObject.SetActive(true);
            }
        }

#if UNITY_EDITOR
        private void OnValidate() {
            var ar = SourceImage.GetComponent<AspectRatioFitter>();
            if (SampleImage != null && ar != null) {
                ar.aspectRatio = (float)SampleImage.width / SampleImage.height;
                SourceImage.texture = SampleImage;
            }
        }
#endif
    }
}