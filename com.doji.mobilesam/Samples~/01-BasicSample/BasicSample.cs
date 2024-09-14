using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Doji.AI.Segmentation.Samples {

    public class MobileSAM_BasicSample : MonoBehaviour, IPointerDownHandler {

        private MobileSAM _mobileSAMPredictor;

        public Texture2D SampleImage;
        public RawImage SourceImage;
        public RawImage MaskImage;
        public GameObject DebugMarkerPrefab;
        public RenderTexture Result;

        private List<float> _points = new List<float>();
        private List<float> _labels = new List<float>();
        
        private List<GameObject> _pointUIElements = new List<GameObject>();

        public void Start () {
            _mobileSAMPredictor = new MobileSAM();
            _mobileSAMPredictor.SetImage(SourceImage.texture);
            Result = _mobileSAMPredictor.Result;
            MaskImage.texture = _mobileSAMPredictor.Result;
        }

        private void OnDestroy() {
            _mobileSAMPredictor?.Dispose();
        }

        public void PredictMask() {
            _mobileSAMPredictor.Predict(_points.ToArray(), _labels.ToArray());
            MaskImage.enabled = true;
        }
      
        public void OnPointerDown(PointerEventData eventData) {
            if (eventData.button != PointerEventData.InputButton.Middle) {
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
                    bool foreground = eventData.button == PointerEventData.InputButton.Left;
                    AddDebugMarker(uvPosition, foreground);
                    AddPoint(textureSpacePos, foreground);
                    PredictMask();
                }
            }
        }

        private void AddPoint(Vector2 point, bool foreground) {
            _points.Add(point.x);
            _points.Add(point.y);
            _labels.Add(foreground ? 1f : 0f);
        }

        private void AddDebugMarker(Vector2 uvPosition, bool foreground) {
            var debugMarker = Instantiate(DebugMarkerPrefab, SourceImage.transform);
            var rT = debugMarker.GetComponent<RectTransform>();
            uvPosition.y = 1f - uvPosition.y;
            rT.anchorMin = uvPosition;
            rT.anchorMax = uvPosition;
            debugMarker.SetActive(true);
            debugMarker.GetComponent<Image>().color = foreground ? Color.green : Color.blue;
            _pointUIElements.Add(debugMarker);
        }

        private void Update() {
            if (Input.GetKeyDown(KeyCode.Escape)) {
                DeleteAllPoints();
            }
        }

        private void DeleteAllPoints() {
            _points.Clear();
            _labels.Clear();
            foreach (var point in _pointUIElements) {
                DestroyImmediate(point);
            }
            _pointUIElements.Clear();
            MaskImage.enabled = false;
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