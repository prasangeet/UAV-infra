#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_USER="${SUDO_USER:-${USER:-}}"
if [[ -z "$TARGET_USER" ]]; then
  echo "Could not determine the desktop login user." >&2
  exit 1
fi

echo "CUBIT Inspect — Linux camera access setup"
echo "User: $TARGET_USER"
echo

if ! getent group video >/dev/null; then
  echo "This system has no 'video' group."
  echo "Camera access may be managed by logind/uaccess or a device-specific group."
  echo "Inspect devices with: ls -l /dev/video*"
  exit 2
fi

VIDEO_GROUP="$(getent group video | cut -d: -f1)"
VIDEO_DEVICES=()
shopt -s nullglob
for device in /dev/video*; do
  VIDEO_DEVICES+=("$device")
done

if (("${#VIDEO_DEVICES[@]}" == 0)); then
  echo "No /dev/video* device nodes currently exist."
  echo "Connect/enable your webcam and check: v4l2-ctl --list-devices"
  exit 3
fi

echo "Detected video nodes:"
for device in "${VIDEO_DEVICES[@]}"; do
  ls -l "$device"
done
echo

if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -Fxq "$VIDEO_GROUP"; then
  echo "$TARGET_USER is already a member of group '$VIDEO_GROUP'."
  echo "If the API was started before group membership was applied,"
  echo "fully log out of the desktop session and log back in, then restart it."
  exit 0
fi

echo "Your user is not a member of group '$VIDEO_GROUP'."
echo "Where camera nodes are group-owned by '$VIDEO_GROUP', this blocks OpenCV access."
echo "Adding your account to this group grants access to camera devices assigned to that group."
read -r -p "Add '$TARGET_USER' to '$VIDEO_GROUP' using sudo? [y/N] " answer
case "$answer" in
  [yY]|[yY][eE][sS])
    sudo usermod -aG "$VIDEO_GROUP" "$TARGET_USER"
    echo
    echo "Group membership updated."
    echo "IMPORTANT: fully log out of your Linux desktop session and log back in."
    echo "Then restart the FastAPI server and CUBIT Inspect."
    ;;
  *)
    echo "No permission changes were made."
    ;;
esac
