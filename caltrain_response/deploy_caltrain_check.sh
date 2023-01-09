#!/bin/bash
gcloud config set project tylerpersonalprojects
gcloud config set compute/zone us-west2-a

gcloud functions deploy caltrain_check --entry-point main --runtime python38 --trigger-http --allow-unauthenticated --env-vars-file config.yaml --memory=2048MB --timeout=540s

