const API_BASE = "http://localhost:8765";

export async function loadCalibrationProfiles() {
  const response = await fetch(`${API_BASE}/api/calibration/profiles`);
  if (!response.ok) throw new Error("Could not load calibration profiles");
  return response.json();
}

export async function saveCalibrationProfile(profile) {
  const response = await fetch(`${API_BASE}/api/calibration/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  if (!response.ok) throw new Error("Could not save calibration profile");
  return response.json();
}

export async function autoDetectCalibration(imageDataUrl) {
  const response = await fetch(`${API_BASE}/api/calibration/auto-detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl }),
  });
  if (!response.ok) throw new Error("Auto-detect failed");
  return response.json();
}
