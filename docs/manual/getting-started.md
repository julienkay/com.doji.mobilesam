# Getting Started

Welcome to the MobileSAM package! This package allows you to perform segmentation tasks in Unity using MobileSAM models. Below is a quick guide on how to get started with this package, including code samples to help you integrate it into your Unity project.

## Table of Contents
- [Installation](#installation)
- [Basic Setup](#basic-setup)
- [Performing Segmentation](#performing-segmentation)
- [Full Example](#full-example)

## Installation

1. Import the MobileSAM package into your Unity project via OpenUPM.

In `Edit -> Project Settings -> Package Manager`
add a new scoped registry:

`Name: Doji`
`URL: https://package.openupm.com`
`Scope(s): com.doji`

In the Package Manager install `com.doji.mobilesam` either by name or select it in the list under `Package Manager -> My Registries`

## Basic Setup

To begin using MobileSAM, you need to create an instance of the `MobileSAM` class. The instance will handle loading the models and initializing the segmentation process.

```csharp
using Doji.AI.Segmentation;

// Initialize MobileSAM
MobileSAM mobileSAMPredictor = new MobileSAM();
```

## Performing Segmentation

To perform segmentation on an image, you need to provide the input image as a `Texture` and provide point prompts that guide the segmentation process.

### Setting an Image for Segmentation

Before running the segmentation, you must set the image using the @Doji.AI.Segmentation.MobileSAM.SetImage(Texture) method:

```csharp
_mobileSAMPredictor.SetImage(TestImage);
```

This method processes the image and prepares it for segmentation.

### Predicting Masks

To predict masks, use the <xref href="Doji.AI.Segmentation.MobileSAM.Predict(System.Single%5b%5d%2cSystem.Single%5b%5d%2cSystem.Nullable%7bRect%7d%2cTexture)" data-throw-if-not-resolved="false"></xref> method:

```csharp
_mobileSAMPredictor.Predict(pointCoords, pointLabels);
```

- `pointCoords`: An array of point coordinates `[x1, y1, x2, y2, ...]` in pixel values.
- `pointLabels`: An array of labels corresponding to each point (1 for foreground, 0 for background).

**Note**: Currently, the `box` and `maskInput` parameters are not yet supported.

### Retrieving Results

The segmentation result is stored in the `Result` property:

```csharp
RenderTexture resultTexture = _mobileSAMPredictor.Result;
```

You can then use this `RenderTexture` in your Unity application, for example, to display it on a UI element or apply it to a material.

### Cleaning Up

Always dispose of the `MobileSAM` instance when you're done using it to free up resources:

```csharp
_mobileSAMPredictor.Dispose();
```

## Full Example

Here's a complete example of setting up and using the MobileSAM predictor in a Unity script:

```csharp
using UnityEngine;
using Doji.AI.Segmentation;

public class SegmentationExample : MonoBehaviour
{
    private MobileSAM _mobileSAMPredictor;
    public Texture TestImage;
    public RenderTexture Result;

    private void Start()
    {
        // Initialize the MobileSAM predictor
        _mobileSAMPredictor = new MobileSAM();
        
        // Set the image for segmentation
        _mobileSAMPredictor.SetImage(TestImage);
        
        // Perform segmentation
        DoSegmentation();
    }

    private void DoSegmentation()
    {
        if (TestImage == null) {
            Debug.LogError("TestImage is null. Assign a texture to TestImage.");
            return;
        }

        // Example prompt: a single point in the middle of the image
        float[] pointCoords = new float[] { TestImage.width / 2f, TestImage.height / 2f };
        float[] pointLabels = new float[] { 1f };  // 1 for foreground point

        // Perform segmentation
        _mobileSAMPredictor.Predict(pointCoords, pointLabels);

        // Retrieve the result
        Result = _mobileSAMPredictor.Result;
        
        // Use the result texture (e.g., display on UI, apply to a material, etc.)
    }

    private void OnDestroy()
    {
        // Clean up resources
        _mobileSAMPredictor.Dispose();
    }
}
```
