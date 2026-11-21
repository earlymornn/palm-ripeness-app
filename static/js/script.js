
// =====================================
// ELEMENT DETEKSI
// =====================================

const detectForm = document.getElementById("detectForm");
const imageInput = document.getElementById("imageInput");
const modelSelect = document.getElementById("modelSelect");

const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");

const detectButton = document.getElementById("detectButton");
const loadingState = document.getElementById("loadingState");
const errorMessage = document.getElementById("errorMessage");

const resultSection = document.getElementById("resultSection");
const resultImage = document.getElementById("resultImage");
const resultModel = document.getElementById("resultModel");
const resultTotal = document.getElementById("resultTotal");
const resultCounts = document.getElementById("resultCounts");
const detectionTable = document.getElementById("detectionTable");

if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];

        errorMessage.textContent = "";

        if (resultSection) {
            resultSection.style.display = "none";
        }

        if (!file) {
            previewBox.style.display = "none";
            return;
        }

        if (!file.type.startsWith("image/")) {
            errorMessage.textContent = "File harus berupa gambar.";
            imageInput.value = "";
            previewBox.style.display = "none";
            return;
        }

        const imageURL = URL.createObjectURL(file);

        previewImage.src = imageURL;
        previewBox.style.display = "block";
    });
}

if (detectForm) {
    detectForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const file = imageInput.files[0];

        if (!file) {
            errorMessage.textContent = "Silakan pilih gambar terlebih dahulu.";
            return;
        }

        const formData = new FormData();

        formData.append("image", file);
        formData.append("model", modelSelect.value);

        detectButton.disabled = true;
        detectButton.textContent = "Memproses...";
        loadingState.style.display = "block";
        errorMessage.textContent = "";
        resultSection.style.display = "none";

        try {
            const response = await fetch("/detect", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Deteksi gagal.");
            }

            resultImage.src = data.image + "?t=" + Date.now();
            resultModel.textContent =
                "Model yang digunakan: " + data.model;
            resultTotal.textContent = data.total;
            resultCounts.innerHTML = "";

            const classLabels = {
                unripe: "Unripe",
                ripe: "Ripe",
                overripe: "Overripe",
                empty: "Empty"
            };

            for (const className in data.counts) {
                const card = document.createElement("div");
                card.className = "count-card";

                const number = document.createElement("strong");
                number.textContent = data.counts[className];

                const label = document.createElement("span");
                label.textContent =
                    classLabels[className] || className;

                card.appendChild(number);
                card.appendChild(label);

                resultCounts.appendChild(card);
            }

            detectionTable.innerHTML = "";

            if (data.detections.length === 0) {
                detectionTable.textContent =
                    "Tidak ada objek yang terdeteksi.";
            } else {
                data.detections.forEach((detection, index) => {
                    const item = document.createElement("div");
                    item.className = "detection-item";

                    const name = document.createElement("span");
                    name.textContent =
                        `${index + 1}. ${classLabels[detection.class] || detection.class}`;

                    const confidence = document.createElement("strong");
                    confidence.textContent =
                        detection.confidence + "%";

                    item.appendChild(name);
                    item.appendChild(confidence);

                    detectionTable.appendChild(item);
                });
            }

            resultSection.style.display = "block";

            resultSection.scrollIntoView({
                behavior: "smooth"
            });

        } catch (error) {
            errorMessage.textContent = error.message;

        } finally {
            detectButton.disabled = false;
            detectButton.textContent = "Deteksi Sekarang";
            loadingState.style.display = "none";
        }
    });
}

const progressBars = document.querySelectorAll(".progress-fill");

progressBars.forEach(function (bar) {
    const percentage = Number(bar.dataset.percentage) || 0;

    bar.style.width = percentage + "%";
});
