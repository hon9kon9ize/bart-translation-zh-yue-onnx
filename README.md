# Bart model with onnxruntime deploying to Google Cloud Run

This repository contains the code for deploying a [ONNX bart translation model](https://huggingface.co/hon9kon9ize/bart-translation-zh-yue-onnx) to Google Cloud Run, It is based on the insights shared in the blog post ['My Journey to a serverless transformers pipeline on Google Cloud'](https://huggingface.co/blog/how-to-deploy-a-pipeline-to-google-clouds) from the Hugging Face website."


## Prerequisites
- Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Install docker

## Deploying to Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/<project-id>/bart-translation-zh-yue-onnx
gcloud run deploy --image gcr.io/<project-id>/bart-translation-zh-yue-onnx --platform managed \
  --command "serve" \
  --cpu 4 \
  --memory 8Gi \
```
