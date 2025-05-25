# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biomechanics tracking workout app with real-time camera-based body tracking, bidirectional audio streaming, and on-device ML processing. The current files are Apple's boilerplate and will be replaced.

## Required Frameworks

**Core Frameworks**:
- `AVFoundation` - Camera capture and audio processing
- `Vision` - Body pose detection and computer vision
- `CoreML` - On-device ML model inference  
- `VideoToolbox` - Hardware-accelerated video processing
- `Metal` - GPU rendering and processing
- `Network` - Modern networking for real-time streaming

**Third-Party Dependencies** (add via SPM):
- WebRTC framework for real-time audio streaming
- Starscream for WebSocket connections
- SwiftyJSON for API communication

## Development Commands

**Device Testing** (Required for camera/audio):
```bash
# Build for connected device
xcodebuild -workspace Movement.xcworkspace -scheme Movement -destination 'platform=iOS,name=iPhone' build

# Install on device
xcrun devicectl device install app Movement.app

# View real-time logs
xcrun devicectl device log show --device-id <DEVICE_ID> --predicate 'subsystem CONTAINS "Movement"'
```

**Performance Profiling**:
```bash
# Time profiler for real-time processing optimization
instruments -t "Time Profiler" -D trace.trace Movement.app

# GPU performance analysis
instruments -t "Metal System Trace" Movement.app
```

**ML Model Integration**:
```bash
# Convert models to CoreML
coremltools-convert --source tensorflow --output BodyPoseModel.mlmodel

# Optimize for Neural Engine
xcrun coremlcompiler compile BodyPoseModel.mlmodel ModelPackage.mlmodelc
```

## Architecture Patterns

**Real-time Processing Pipeline**:
- Separate dispatch queues for camera, audio, ML, and network processing
- CVPixelBuffer pooling to minimize memory allocations
- 60 FPS target with 16.67ms frame processing budget

**Threading Architecture**:
```swift
private let cameraQueue = DispatchQueue(label: "camera", qos: .userInitiated)
private let audioQueue = DispatchQueue(label: "audio", qos: .userInitiated)  
private let mlQueue = DispatchQueue(label: "ml", qos: .userInitiated)
private let networkQueue = DispatchQueue(label: "network", qos: .userInitiated)
```

**Core Managers**:
- `CameraManager` - AVCaptureSession with real-time frame processing
- `AudioStreamManager` - Bidirectional audio streaming via AVAudioEngine
- `BodyTrackingManager` - Vision framework body pose detection
- `NetworkManager` - WebRTC/WebSocket real-time communication

## Required Permissions

**Info.plist**:
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access required for biomechanics tracking</string>
<key>NSMicrophoneUsageDescription</key>
<string>Microphone access required for audio coaching</string>
<key>NSLocalNetworkUsageDescription</key>
<string>Network access required for real-time coaching</string>
```

**Entitlements** (replace existing sandbox):
- `com.apple.security.camera`
- `com.apple.security.microphone` 
- `com.apple.security.network.client`
- `com.apple.developer.networking.wifi-info`

## SwiftUI Implementation

**Main Screen**: Front-facing camera view with overlay UI
**Bottom Sheets**: Use `.sheet()` with `.presentationDetents()` for NUX, permissions, info
**Real-time UI Updates**: Use `@ObservableObject` managers with `@Published` properties

## Performance Considerations

- Target <20ms audio latency for real-time coaching
- Use Metal Performance Shaders for GPU acceleration
- Implement adaptive quality based on device thermal state
- Memory management with autoreleasepool for video processing loops
- Monitor with os_signpost for optimization profiling