export function drawPitchMap(canvas, results) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b0d0d";
  ctx.fillRect(0, 0, width, height);

  const pitch = { x: 70, y: 20, w: width - 140, h: height - 48 };
  ctx.fillStyle = "#8f7d55";
  ctx.fillRect(pitch.x, pitch.y, pitch.w, pitch.h);
  ctx.strokeStyle = "#e8e1c4";
  ctx.lineWidth = 2;
  ctx.strokeRect(pitch.x, pitch.y, pitch.w, pitch.h);

  const stumpY = pitch.y + pitch.h - 22;
  ctx.strokeStyle = "#f4e8be";
  [-12, 0, 12].forEach((offset) => {
    ctx.beginPath();
    ctx.moveTo(width / 2 + offset, stumpY - 24);
    ctx.lineTo(width / 2 + offset, stumpY + 12);
    ctx.stroke();
  });

  const points = results?.trajectory?.points || results?.trajectory?.path || [];
  const mapped = points.map((point, index) => ({
    x: pitch.x + 20 + (index / Math.max(1, points.length - 1)) * (pitch.w - 40),
    y: pitch.y + pitch.h - 28 - Number(point.z || 0) * 80 - Number(point.y || 0) * 8,
  }));
  if (mapped.length > 1) {
    ctx.strokeStyle = "#00bcd4";
    ctx.lineWidth = 3;
    ctx.beginPath();
    mapped.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }

  const bounce = mapped[Math.max(0, Math.floor(mapped.length * 0.45))];
  if (bounce) {
    ctx.fillStyle = "#ff9800";
    ctx.beginPath();
    ctx.arc(bounce.x, bounce.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  const impact = mapped[Math.max(0, Math.floor(mapped.length * 0.75))];
  if (impact) {
    ctx.strokeStyle = "#f44336";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(impact.x - 7, impact.y - 7);
    ctx.lineTo(impact.x + 7, impact.y + 7);
    ctx.moveTo(impact.x + 7, impact.y - 7);
    ctx.lineTo(impact.x - 7, impact.y + 7);
    ctx.stroke();
  }

  ctx.fillStyle = "#aaa";
  ctx.font = "12px Consolas, monospace";
  ctx.fillText("Scale: 20.12m pitch", 12, height - 12);
}
