/*!
 * Copyright (c) 2024 by Contributors
 * \file backend_config.cpp
 * \brief Implementation of global backend configuration
 */
#include "pypto/backend/backend_config.h"

#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

std::optional<BackendType> BackendConfig::backend_type_;
std::mutex BackendConfig::mutex_;

void BackendConfig::SetBackendType(BackendType type) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (backend_type_.has_value()) {
    // Idempotent: allow setting the same type multiple times
    CHECK(*backend_type_ == type)
        << "Backend type already set to "
        << (*backend_type_ == BackendType::CCE ? "CCE" : "PTO")
        << ", cannot change to " << (type == BackendType::CCE ? "CCE" : "PTO");
    return;
  }

  backend_type_ = type;
}

const Backend* BackendConfig::GetBackend() {
  std::lock_guard<std::mutex> lock(mutex_);

  CHECK(backend_type_.has_value())
      << "Backend type not configured. "
      << "Please call SetBackendType() or use compile() with backend_type parameter.";

  return GetBackendInstance(*backend_type_);
}

BackendType BackendConfig::GetBackendType() {
  std::lock_guard<std::mutex> lock(mutex_);

  CHECK(backend_type_.has_value())
      << "Backend type not configured. "
      << "Please call SetBackendType() or use compile() with backend_type parameter.";

  return *backend_type_;
}

bool BackendConfig::IsConfigured() {
  std::lock_guard<std::mutex> lock(mutex_);
  return backend_type_.has_value();
}

void BackendConfig::ResetForTesting() {
  std::lock_guard<std::mutex> lock(mutex_);
  backend_type_.reset();
}

}  // namespace backend
}  // namespace pypto
