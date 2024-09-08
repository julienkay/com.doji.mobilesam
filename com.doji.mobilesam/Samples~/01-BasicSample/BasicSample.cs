using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Doji.AI.Segmentation.Samples {

    public class MobileSAM_BasicSample : MonoBehaviour, IPointerClickHandler {

        private MobileSAM _mobileSAMPredictor;

        public RawImage SourceImage;
        public RawImage MaskImage;
        public RectTransform DebugMarker;
        public RenderTexture Result;

        private Vector2? _currentPoint;
        private Camera _eventCamera;

        public void Start () {
            _mobileSAMPredictor = new MobileSAM();
            Result = _mobileSAMPredictor.Result;
            MaskImage.texture = _mobileSAMPredictor.Result;
        }

        private void OnDestroy() {
            _mobileSAMPredictor?.Dispose();
        }

        public void PredictMask(Vector2 point) {
            //point *= new Vector2(1f, -1f);
            //Debug.Log(point);
            var points = new float[] { point.x, point.y };
            var labels = new float[] { 1f /* foreground */ };
            _mobileSAMPredictor.PredictMasks(SourceImage.texture, points, labels);
            MaskImage.enabled = true;
        }
      
        public void OnPointerClick(PointerEventData eventData) {
            if (eventData.button == PointerEventData.InputButton.Left) {
                RectTransform rectTransform = SourceImage.GetComponent<RectTransform>();
                _currentPoint = eventData.position;
                _eventCamera = eventData.pressEventCamera;

                Vector2 localPoint;
                RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, eventData.position, eventData.pressEventCamera, out localPoint);

                Rect rect = rectTransform.rect;
                float normalizedX = (localPoint.x - rect.x) / rect.width;
                float normalizedY = (localPoint.y - rect.y) / rect.height;

                Vector2 uvPosition = new Vector2(normalizedX, 1.0f - normalizedY);
                Vector2 textureSpacePos = new Vector2(SourceImage.texture.width, SourceImage.texture.height) * uvPosition;
                Debug.Log(textureSpacePos);

                // Ensure click was inside of the texture area
                if (uvPosition.x >= 0 && uvPosition.x <= 1 && uvPosition.y >= 0 && uvPosition.y <= 1) {
                    MoveDebugMarker(localPoint);
                    PredictMask(textureSpacePos);
                }
            }
        }

        private void OnRectTransformDimensionsChange() {
            // move point in case canvas changed size
            if (_currentPoint != null) {
                RectTransform rectTransform = SourceImage.GetComponent<RectTransform>();
                Vector2 localPoint;
                RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, _currentPoint.Value, _eventCamera, out localPoint);
                MoveDebugMarker(localPoint);
            }
        }

        private void MoveDebugMarker(Vector2 localPosition) {
            if (DebugMarker != null) {
                DebugMarker.anchoredPosition = localPosition;
                DebugMarker.gameObject.SetActive(true);
            }
        }
    }
}