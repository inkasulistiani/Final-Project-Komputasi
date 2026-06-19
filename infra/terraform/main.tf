# ============================================================
# terraform/main.tf — Provisioning GCP Infrastructure
# ============================================================
# Resources yang dibuat:
#   1. VPC Network + Subnet
#   2. Firewall rules (SSH, Spark UI, Prometheus, Grafana)
#   3. GCS Bucket (dataset + output)
#   4. Dataproc Cluster (1 master + 3 worker)
#   5. Compute Engine VM untuk CPU/OpenMP testing
#
# Cara pakai:
#   cd infra/terraform
#   terraform init
#   terraform plan -var="project_id=YOUR_PROJECT"
#   terraform apply -var="project_id=YOUR_PROJECT"
#   terraform destroy  # cleanup (PENTING: hentikan billing)
# ============================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  # Simpan state di GCS agar bisa diakses tim
  backend "gcs" {
    bucket = "kmeans-tf-state"
    prefix = "terraform/state"
  }
}

# ── Provider ────────────────────────────────────────────────
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── Variables ────────────────────────────────────────────────
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}
variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}
variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}
variable "cluster_name" {
  description = "Nama Dataproc cluster"
  type        = string
  default     = "kmeans-cluster"
}
variable "worker_count" {
  description = "Jumlah worker node Dataproc"
  type        = number
  default     = 3
}
variable "worker_machine_type" {
  description = "Machine type worker (misal: n1-standard-4)"
  type        = string
  default     = "n1-standard-4"   # 4 vCPU, 15 GB RAM
}
variable "gpu_machine_type" {
  description = "Machine type untuk CUDA testing"
  type        = string
  default     = "n1-standard-4"
}

# ── Local values ─────────────────────────────────────────────
locals {
  bucket_name  = "${var.project_id}-kmeans-data"
  labels       = { project = "kmeans", env = "experiment" }
}

# ── 1. VPC Network ──────────────────────────────────────────
resource "google_compute_network" "kmeans_vpc" {
  name                    = "kmeans-vpc"
  auto_create_subnetworks = false
  description             = "VPC untuk kmeans experiment"
}

resource "google_compute_subnetwork" "kmeans_subnet" {
  name          = "kmeans-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.kmeans_vpc.id

  # Flow logs untuk monitoring traffic antar node
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# ── 2. Firewall Rules ────────────────────────────────────────
# SSH access
resource "google_compute_firewall" "allow_ssh" {
  name    = "kmeans-allow-ssh"
  network = google_compute_network.kmeans_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]   # Ganti dengan IP kantor di produksi!
  target_tags   = ["kmeans-node"]
}

# Spark UI + internal communication
resource "google_compute_firewall" "allow_spark" {
  name    = "kmeans-allow-spark"
  network = google_compute_network.kmeans_vpc.name

  allow {
    protocol = "tcp"
    ports    = [
      "7077",    # Spark Master
      "8080",    # Spark Master Web UI
      "8081",    # Spark Worker Web UI
      "4040",    # Spark App UI
      "18080",   # Spark History Server
    ]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["kmeans-node"]
}

# Monitoring (Prometheus + Grafana)
resource "google_compute_firewall" "allow_monitoring" {
  name    = "kmeans-allow-monitoring"
  network = google_compute_network.kmeans_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["9090", "3000", "9100", "9404", "9405", "9406"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["kmeans-node"]
}

# Internal traffic (semua port antar node)
resource "google_compute_firewall" "allow_internal" {
  name    = "kmeans-allow-internal"
  network = google_compute_network.kmeans_vpc.name

  allow { protocol = "tcp"  }
  allow { protocol = "udp"  }
  allow { protocol = "icmp" }
  source_ranges = ["10.10.0.0/24"]
}

# ── 3. GCS Bucket (Data Storage) ────────────────────────────
resource "google_storage_bucket" "data_bucket" {
  name          = local.bucket_name
  location      = var.region
  force_destroy = true   # Izinkan destroy meski ada file (eksperimen only)

  # Lifecycle: hapus file output setelah 30 hari (hemat biaya)
  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 30 }
  }

  uniform_bucket_level_access = true
  labels                      = local.labels
}

# Folder struktur di GCS
resource "google_storage_bucket_object" "data_folder" {
  name    = "data/"
  content = " "
  bucket  = google_storage_bucket.data_bucket.name
}
resource "google_storage_bucket_object" "results_folder" {
  name    = "results/"
  content = " "
  bucket  = google_storage_bucket.data_bucket.name
}
resource "google_storage_bucket_object" "logs_folder" {
  name    = "logs/"
  content = " "
  bucket  = google_storage_bucket.data_bucket.name
}

# ── 4. Dataproc Cluster (Distributed Spark) ─────────────────
resource "google_dataproc_cluster" "kmeans_cluster" {
  name    = var.cluster_name
  region  = var.region
  labels  = local.labels

  cluster_config {
    staging_bucket = google_storage_bucket.data_bucket.name

    # Master node (driver)
    master_config {
      num_instances = 1
      machine_type  = "n1-standard-4"  # 4 vCPU, 15 GB RAM

      disk_config {
        boot_disk_type    = "pd-ssd"
        boot_disk_size_gb = 100
      }
    }

    # Worker nodes (executors)
    worker_config {
      num_instances = var.worker_count
      machine_type  = var.worker_machine_type

      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 200   # Cukup untuk spill ke disk
        num_local_ssds    = 1     # SSD lokal untuk shuffle cepat
      }
    }

    # Preemptible workers (hemat biaya ~80%, bisa dihapus kapan saja)
    # Cocok untuk eksperimen non-kritis
    preemptible_worker_config {
      num_instances = 0   # Set > 0 untuk hemat biaya
    }

    # Software (Spark + Hadoop)
    software_config {
      image_version = "2.1-ubuntu20"   # Spark 3.3, Hadoop 3.3
      override_properties = {
        "spark:spark.executor.memory"        = "3g"
        "spark:spark.executor.cores"         = "2"
        "spark:spark.sql.adaptive.enabled"   = "true"
        "spark:spark.eventLog.enabled"       = "true"
        "spark:spark.eventLog.dir"           = "gs://${local.bucket_name}/logs/spark-events"
        "spark:spark.history.fs.logDirectory"= "gs://${local.bucket_name}/logs/spark-events"
        # Prometheus JMX exporter
        "spark:spark.driver.extraJavaOptions" = "-javaagent:/opt/jmx_exporter.jar=9090:/opt/spark-jmx.yml"
      }
    }

    # Jaringan
    gce_cluster_config {
      subnetwork       = google_compute_subnetwork.kmeans_subnet.id
      tags             = ["kmeans-node"]
      service_account_scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
      ]
      # Tidak pakai IP publik untuk worker (lebih aman, hemat biaya)
      internal_ip_only = false
    }

    # Inisialisasi: install Python deps di semua node
    initialization_action {
      script      = "gs://${local.bucket_name}/scripts/init_cluster.sh"
      timeout_sec = 300
    }
  }
}

# ── 5. VM untuk CPU/OpenMP Testing ──────────────────────────
resource "google_compute_instance" "cpu_vm" {
  name         = "kmeans-cpu-vm"
  machine_type = "c2-standard-16"   # 16 vCPU (compute-optimized, ideal OpenMP)
  zone         = var.zone
  tags         = ["kmeans-node"]
  labels       = local.labels

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.kmeans_subnet.id
    access_config { }   # Ephemeral public IP
  }

  # Script startup: install GCC + OpenMP, download dataset
  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e
    apt-get update -q
    apt-get install -y -q build-essential g++-12 libomp-dev

    # Download dataset dari GCS
    apt-get install -y -q google-cloud-sdk
    gsutil cp gs://${local.bucket_name}/data/nyc_taxi.csv /data/nyc_taxi.csv

    # Compile kode
    g++ -O3 -std=c++17 -fopenmp -march=native \
        -o /usr/local/bin/kmeans_omp \
        /app/src/openmp/kmeans_omp.cpp
    echo "Setup selesai" > /var/log/kmeans_setup.log
  EOF

  service_account {
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

# ── 6. VM untuk GPU/CUDA Testing ────────────────────────────
resource "google_compute_instance" "gpu_vm" {
  name         = "kmeans-gpu-vm"
  machine_type = var.gpu_machine_type
  zone         = var.zone
  tags         = ["kmeans-node"]
  labels       = local.labels

  boot_disk {
    initialize_params {
      image = "projects/ml-images/global/images/c0-deeplearning-common-gpu-v20231209-debian-11"
      size  = 100
      type  = "pd-ssd"
    }
  }

  # Attach NVIDIA T4 GPU
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  # Wajib untuk VM dengan GPU
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    subnetwork = google_compute_subnetwork.kmeans_subnet.id
    access_config { }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e
    # Driver NVIDIA sudah pre-installed di deep learning image
    # Install CUDA toolkit tambahan jika diperlukan
    apt-get update -q
    apt-get install -y -q cuda-toolkit-12-2

    # Download dataset
    gsutil cp gs://${local.bucket_name}/data/nyc_taxi.csv /data/nyc_taxi.csv
    echo "GPU setup selesai" > /var/log/kmeans_gpu_setup.log
  EOF

  service_account {
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

# ── Outputs ──────────────────────────────────────────────────
output "dataproc_master_ip" {
  description = "IP address Dataproc master node"
  value       = google_dataproc_cluster.kmeans_cluster.cluster_config[0].master_config[0].instance_names[0]
}
output "cpu_vm_external_ip" {
  description = "IP external VM CPU"
  value       = google_compute_instance.cpu_vm.network_interface[0].access_config[0].nat_ip
}
output "gpu_vm_external_ip" {
  description = "IP external VM GPU"
  value       = google_compute_instance.gpu_vm.network_interface[0].access_config[0].nat_ip
}
output "gcs_bucket" {
  description = "GCS bucket untuk data dan output"
  value       = google_storage_bucket.data_bucket.url
}
output "spark_master_ui" {
  description = "URL Spark Master Web UI"
  value       = "http://${google_compute_instance.cpu_vm.network_interface[0].access_config[0].nat_ip}:8080"
}
