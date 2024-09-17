/// Copyright (c) 2024 Kodeco Inc.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
///
/// This project and source code may use libraries or frameworks that are
/// released under various Open-Source licenses. Use of those libraries and
/// frameworks are governed by their own individual licenses.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import SwiftUI
import PhotosUI
import Vision

struct ContentView: View {
  @State private var selectedImage: PhotosPickerItem?
  @State private var image: Image?
  @State private var cgImage: CGImage?
  @State private var detectedObjects: [DetectedObject] = []

  func runModel() {
    guard
      // 1
      let cgImage = cgImage,
      // 2
      // NOTE: you will need to import the model files in Lesson 3, Instruction 1 for this project to compile. We aren't included 
      let model = try? yolov8x_oiv7(configuration: .init()).model,
      // 3
      let detector = try? VNCoreMLModel(for: model) else {
        // 4
        print("Unable to load photo.")
        return
    }
    // 1
    let visionRequest = VNCoreMLRequest(model: detector) { request, error in
      detectedObjects = []
      if let error = error {
        print(error.localizedDescription)
        return
      }
      // 2
      if let results = request.results as? [VNRecognizedObjectObservation] {
        // 1
        if results.isEmpty {
          print("No results found.")
          return
        }
        // 2
        for result in results {
          // 3
          if let firstIdentifier = result.labels.first {
            let confidence = firstIdentifier.confidence
            let label = firstIdentifier.identifier
            // 4
            let boundingBox = result.boundingBox
            // 5
            let object = DetectedObject(
              label: label,
              confidence: confidence,
              boundingBox: boundingBox
            )
            detectedObjects.append(object)
          }
        }
      }
    }
    // 1
    visionRequest.imageCropAndScaleOption = .scaleFill
    // 2
    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
    // 3
    do {
      try handler.perform([visionRequest])
    } catch {
      print(error)
    }
  }
  
  var body: some View {
    PhotosPicker("Select Photo", selection: $selectedImage, matching: .images)
      .onChange(of: selectedImage) {
        Task {
          if
            let loadedImageData = try? await selectedImage?.loadTransferable(type: Data.self),
            let uiImage = UIImage(data: loadedImageData) {
            image = Image(uiImage: uiImage)
            cgImage = uiImage.cgImage
          }
        }
      }
      .onChange(of: cgImage) {
        runModel()
      }
    if let image = image {
      ImageDisplayView(image: image)
        .overlay {
          ForEach(detectedObjects, id: \.self) { ident in
            `ObjectOverlayView`(object: ident)
          }
        }
    } else {
      NoImageSelectedView()
    }
    ForEach(detectedObjects, id: \.self) { obj in
      Text(obj.label) + Text(" (") + Text(obj.confidence, format: .percent) + Text(")")
    }
  }
}

#Preview {
  ContentView()
}
