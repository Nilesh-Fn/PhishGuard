// Animate prediction and confidence bar
window.addEventListener('DOMContentLoaded', (event) => {
  const predictionDiv = document.querySelector('.prediction');
  const confidenceBar = document.querySelector('.confidence-bar');

  if (predictionDiv) {
    // Show prediction with fade-in
    setTimeout(() => {
      predictionDiv.classList.add('show');
    }, 200);

    // Animate confidence bar
    if (confidenceBar) {
      const width = confidenceBar.getAttribute('data-confidence');
      confidenceBar.style.width = width + '%';
    }
  }
});
