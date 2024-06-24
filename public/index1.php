<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition with face-api.js</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        canvas, video {
            position: absolute;
        }
    </style>
</head>
<body>
    <video id="video" width="720" height="560" autoplay muted></video>
    <canvas id="overlay" width="720" height="560"></canvas>
    <script defer src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api@latest/dist/face-api.min.js"></script>

    <script>
        document.addEventListener('DOMContentLoaded', async () => {
            const video = document.getElementById('video');

            // Load models
            await Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri('models'),
                faceapi.nets.faceLandmark68Net.loadFromUri('models'),
                faceapi.nets.faceRecognitionNet.loadFromUri('models'),
                faceapi.nets.faceExpressionNet.loadFromUri('models')
            ]);

            async function startVideo() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
                    video.srcObject = stream;
                } catch (err) {
                    console.error('Error accessing webcam:', err);
                }
            }

            startVideo();

            video.addEventListener('play', async () => {
                const canvas = document.getElementById('overlay');
                const displaySize = { width: video.width, height: video.height };
                faceapi.matchDimensions(canvas, displaySize);

                const labeledFaceDescriptors = await loadLabeledImages();
                const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

                setInterval(async () => {
                    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                        .withFaceLandmarks()
                        .withFaceDescriptors();
                    const resizedDetections = faceapi.resizeResults(detections, displaySize);

                    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                    faceapi.draw.drawDetections(canvas, resizedDetections);
                    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

                    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
                    results.forEach((result, i) => {
                        const box = resizedDetections[i].detection.box;
                        const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
                        drawBox.draw(canvas);
                    });
                }, 100);
            });

            async function loadLabeledImages() {
                const label = 'riskhan'; // Ganti dengan label orang yang ingin dikenali
                const descriptions = [];

                for (let i = 1; i <= 3; i++) {
                    const img = await faceapi.fetchImage(`labeled_images/${label}/${i}.jpg`);
                    const detections = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
                    descriptions.push(detections.descriptor);
                }

                return [new faceapi.LabeledFaceDescriptors(label, descriptions)];
            }
        });
    </script>
</body>
</html>
