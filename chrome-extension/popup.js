const API_URL = 'http://localhost:8000/predict';

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const preview = document.getElementById('preview');
const result = document.getElementById('result');
const loading = document.getElementById('loading');
const error = document.getElementById('error');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = '#4CAF50';
});
uploadArea.addEventListener('dragleave', () => {
  uploadArea.style.borderColor = '#ccc';
});
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = '#ccc';
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Veuillez sélectionner une image');
    return;
  }

  hideAll();
  showLoading();

  const reader = new FileReader();
  reader.onload = async (e) => {
    const imageData = e.target.result;
    showPreview(imageData);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Erreur: ${response.statusText}`);
      }

      const data = await response.json();
      showResult(data);
    } catch (err) {
      showError(`Erreur: ${err.message}. Assurez-vous que l'API est démarrée sur ${API_URL}`);
    }
  };

  reader.readAsDataURL(file);
}

function showPreview(imageData) {
  preview.innerHTML = `<img src="${imageData}" alt="Preview">`;
  preview.classList.remove('hidden');
}

function showResult(data) {
  hideAll();
  result.className = `result ${data.label}`;
  result.innerHTML = `
    <div>${data.label === 'fake' ? '❌ FAKE' : '✅ REAL'}</div>
    <div class="confidence">Confiance: ${data.confidence}%</div>
    <div class="confidence" style="font-size: 12px; margin-top: 5px;">
      Real: ${data.real_confidence}% | Fake: ${data.fake_confidence}%
    </div>
  `;
  result.classList.remove('hidden');
}

function showLoading() {
  loading.classList.remove('hidden');
}

function showError(message) {
  hideAll();
  error.textContent = message;
  error.classList.remove('hidden');
}

function hideAll() {
  preview.classList.add('hidden');
  result.classList.add('hidden');
  loading.classList.add('hidden');
  error.classList.add('hidden');
}

