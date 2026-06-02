# AWS Event-Driven PDF RAG Comparison App

This folder contains the  starter code for the AWS version of the PDF RAG
comparison app. You are expected to create and connect the required AWS resources 
yourself, then run this code on your web and worker machines.

The application uses this flow:

1. The Flask web app uploads a PDF to S3 and records document metadata in
   PostgreSQL.
3. The upload event is fanned out to two SQS queues. 
4. One worker consumes the fixed-size queue and chunks the PDF with fixed-size
   chunks.
5. Another worker consumes the paragraph-aware queue and chunks the same PDF by
   paragraph boundaries.
6. The web app reads the processing results from PostgreSQL and lets you compare
   retrieval results for both strategies.

## Files

- `app.py`: Flask web application. It handles PDF uploads, writes document
  metadata to PostgreSQL, displays processing status, and compares retrieval
  results after both workers finish.
- `worker_fixed_size.py`: Starts a worker for the fixed-size chunking strategy.
  It reads SQS messages from `FIXED_SIZE_QUEUE_URL`.
- `worker_paragraph_aware.py`: Starts a worker for the paragraph-aware chunking
  strategy. It reads SQS messages from `PARAGRAPH_AWARE_QUEUE_URL`.
- `worker_common.py`: Shared worker logic. It receives SQS messages, parses S3
  event notifications, downloads PDFs from S3, extracts text, writes chunks to
  PostgreSQL, updates processing status, and deletes successfully processed SQS
  messages.
- `rag.py`: PDF and retrieval helper functions. It validates PDF filenames,
  downloads PDFs, extracts text with `pypdf`, creates chunks, and ranks chunks
  with TF-IDF cosine similarity.
- `db.py`: PostgreSQL helper functions. It opens database connections, creates
  tables from `schema.sql`, creates document rows, creates processing runs, and
  updates document status.
- `schema.sql`: Database schema for documents, processing runs, chunks,
  retrieval queries, and retrieval results.
- `aws_clients.py`: Shared `boto3` S3 and SQS clients configured with
  `AWS_REGION`.
- `config.py`: Reads required configuration from environment variables and
  defines strategy names, file upload limits, and SQS settings.
- `requirements.txt`: Python dependencies required by the web app and workers.


## Environment Variables

Set these environment variables before running the web app or workers:

```sh
export AWS_REGION=us-east-1
export S3_BUCKET_NAME=your-s3-bucket-name
export FIXED_SIZE_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/YOUR_ACCOUNT/fixed-size-queue
export PARAGRAPH_AWARE_QUEUE_URL=https://us-east-1.amazonaws.com/YOUR_ACCOUNT/paragraph-aware-queue
export DATABASE_URL='postgresql://app_user:password@your-rds-endpoint.ap-southeast-2.rds.amazonaws.com:5432/pdf_rag'
```

Optional settings:

```sh
export MAX_FILE_SIZE_MB=5
export SQS_WAIT_TIME_SECONDS=20
export SQS_VISIBILITY_TIMEOUT=300
```

`app.py` requires `S3_BUCKET_NAME` and `DATABASE_URL`. The worker scripts need
all required variables because they need both S3 and SQS access.

## Install Dependencies
On EC2, copy the code to the instance first. 
Run this on each machine where you will run the web app or workers:

```sh
python3 -m pip install -r requirements.txt
```


## Run the Web App

On the web machine:

```sh
source your-env-file.sh
python3 app.py
```

The app listens on port `5000`:

```text
http://WEB_INSTANCE_PUBLIC_IP:5000
```

When `app.py` starts, it calls `init_db()` and creates the required tables if
they do not already exist.

## Run the Workers

On the worker machine, start each worker in a separate terminal session.

Fixed-size worker:

```sh

source your-env-file.sh
python3 -u worker_fixed_size.py
```

Paragraph-aware worker:

```sh
source your-env-file.sh
python3 -u worker_paragraph_aware.py
```

The workers run continuously. Leave both terminals open while testing.

## Test the Application

1. Open the web app in your browser.
2. Upload a PDF.
3. Confirm that the PDF appears under `uploads/` in your S3 bucket.
4. Watch both worker terminals for messages such as:

```text
received SQS message
loading document for s3://...
created N chunks
completed document_id=...
```

5. Refresh or wait for the web page to auto-refresh.
6. Open the document page after both processing runs complete.
7. Enter a query to compare the retrieved chunks from both strategies.

Useful AWS CLI checks:

```sh
aws s3 ls s3://$S3_BUCKET_NAME/uploads/
aws sqs get-queue-attributes --queue-url "$FIXED_SIZE_QUEUE_URL" --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
aws sqs get-queue-attributes --queue-url "$PARAGRAPH_AWARE_QUEUE_URL" --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
```

## Troubleshooting

- If uploads fail, check `S3_BUCKET_NAME`, AWS credentials or instance role
  permissions, and the S3 bucket policy.
- If workers wait forever, check the S3 notification, SNS topic, SNS-to-SQS
  subscriptions, SQS queue policies, and queue URLs.
- If the document appears but never becomes ready, check both worker terminals
  for errors and confirm each worker is connected to the correct queue.
- If database errors occur, check `DATABASE_URL`, database security group rules,
  username/password, and whether the database is reachable from the web and
  worker machines.
- If you want to clear the demo database rows, use the web app's reset button.
  This does not delete uploaded objects from S3.
