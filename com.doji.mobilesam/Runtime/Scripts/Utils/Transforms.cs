using System;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Segmentation {

    public partial class MobileSAM {

        // the input image size of the image encoder
        private const int IMG_SIZE = 1024;

        internal static class Transforms {

            public static Tensor ApplyImage(Texture image) {
                TextureTransform transform = new TextureTransform();

                // expects shape in HxWxC format.
                transform.SetTensorLayout(TensorLayout.NHWC);

                // resize longest side to 1024
                (int newH, int newW) = GetPreprocessShape(image.height, image.width, IMG_SIZE);
                transform.SetDimensions(newW, newH);

                var tensor = TextureConverter.ToTensor(image, transform);
                tensor.Reshape(tensor.shape.Squeeze(0));
                return tensor;
            }

            public static float[] ApplyCoords(float[] pointCoords, (int height, int width) origSize) {
                //TODO: we should pool this
                pointCoords = pointCoords.Copy();
                (float newH, float newW) = GetPreprocessShape(origSize.height, origSize.width, IMG_SIZE);

                for (int i = 0; i < pointCoords.Length; i+=2) {
                    pointCoords[i] = pointCoords[i] * (newW / origSize.width);
                    pointCoords[i+1] = pointCoords[i+1] * (newH / origSize.height);
                }

                return pointCoords;
            }

            private static (int width, int height) GetPreprocessShape(int oldH, int oldW, int longSideLength) {
                float scale = (float)longSideLength / Math.Max(oldW, oldH);
                int newW = (int)(oldW * scale + 0.5f);
                int newH = (int)(oldH * scale + 0.5f);
                return (newH, newW);
            }
        }
    }

    internal static class ArrayUtils {
        public static float[] Copy(this float[] src) {
            float[] target = new float[src.Length];
            src.CopyTo(target, 0);
            return target;
        }
    }
}