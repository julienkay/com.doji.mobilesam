using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Doji.AI.Segmentation.Samples {

    public class BasicSample : MonoBehaviour, IPointerClickHandler {

        private MobileSAM _mobileSAMPredictor;

        public RawImage SourceImage;
        public RectTransform DebugMarker;


        public void Start () {
            _mobileSAMPredictor = new MobileSAM();
        }

        private void OnDestroy() {
            _mobileSAMPredictor?.Dispose();
        }

        public void PredictMask() {
            _mobileSAMPredictor.PredictMasks(SourceImage.texture);
        }
      
        public void OnPointerClick(PointerEventData eventData) {
            if (eventData.button == PointerEventData.InputButton.Left) {
                RectTransform rectTransform = SourceImage.GetComponent<RectTransform>();

                Vector2 localPoint;
                RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, eventData.position, eventData.pressEventCamera, out localPoint);
                
                Rect rect = rectTransform.rect;
                float normalizedX = (localPoint.x - rect.x) / rect.width;
                float normalizedY = (localPoint.y - rect.y) / rect.height;

                Vector2 uvPosition = new Vector2(normalizedX, normalizedY);

                // Ensure click was inside of the texture area
                if (uvPosition.x >= 0 && uvPosition.x <= 1 && uvPosition.y >= 0 && uvPosition.y <= 1) {
                    MoveDebugMarker(localPoint);
                    PredictMask();
                }
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