#!/bin/bash

CERT_DIR="certs"
mkdir -p ${CERT_DIR}
echo "Creating CA..."
openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout "${CERT_DIR}/ca_key.pem" -out "${CERT_DIR}/ca_cert.pem" -subj "/CN=My CA"
echo "Creating server certificate..."
openssl req -newkey rsa:4096 -nodes -keyout "${CERT_DIR}/server_key.pem" -out "${CERT_DIR}/server_csr.pem" -subj "/CN=localhost"
openssl x509 -req -in "${CERT_DIR}/server_csr.pem" -CA "${CERT_DIR}/ca_cert.pem" -CAkey "${CERT_DIR}/ca_key.pem" -CAcreateserial -out "${CERT_DIR}/server_cert.pem" -days 365
echo "Creating client certificate..."
openssl req -newkey rsa:4096 -nodes -keyout "${CERT_DIR}/client_key.pem" -out "${CERT_DIR}/client_csr.pem" -subj "/CN=Client"
openssl x509 -req -in "${CERT_DIR}/client_csr.pem" -CA "${CERT_DIR}/ca_cert.pem" -CAkey "${CERT_DIR}/ca_key.pem" -CAcreateserial -out "${CERT_DIR}/client_cert.pem" -days 365
echo "Certificates generated successfully in ${CERT_DIR}/"
