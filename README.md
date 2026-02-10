# StreamDiffusionV2 Docker Build

Automated Docker build for StreamDiffusionV2 real-time video transformation.

## Usage

The image is automatically built and pushed to:
```
ghcr.io/arubinst/streamdiffusion-v2:latest
```

### Pull and run locally:
```bash
docker pull ghcr.io/arubinst/streamdiffusion-v2:latest
docker run --gpus all -p 7860:7860 ghcr.io/arubinst/streamdiffusion-v2:latest
```

### Use in Kubernetes:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamdiffusion-v2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streamdiffusion
  template:
    metadata:
      labels:
        app: streamdiffusion
    spec:
      containers:
      - name: streamdiffusion
        image: ghcr.io/arubinst/streamdiffusion-v2:latest
        ports:
        - containerPort: 7860
        resources:
          limits:
            nvidia.com/gpu: 1
```

Access the web interface at http://localhost:7860
