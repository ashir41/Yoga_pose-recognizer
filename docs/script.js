import { Client } from "https://esm.sh/@gradio/client";

const input = document.getElementById("photo-input");
const button = document.getElementById("predict-btn");
const results = document.getElementById("results");
const preview = document.getElementById("preview");

let selectedFile = null;

/* Image Preview */
input.addEventListener("change", () => {
    selectedFile = input.files[0];

    if (selectedFile) {
        preview.src = URL.createObjectURL(selectedFile);
        preview.style.display = "block";
        results.innerHTML = "";
    }
});

/* Prediction */
button.addEventListener("click", async () => {

    if (!selectedFile) {
        alert("Please upload an image first!");
        return;
    }

    results.innerHTML = "⏳ Predicting...";

    try {

        const client = await Client.connect("ashir16/yoga_pose_recognizer");

        const result = await client.predict("/predict", {
            image: selectedFile,
        });

        console.log("FULL RESULT:", result);

        // ✅ Direct top label
        const topLabel = result.data[0].label;

        // ✅ Direct top confidence
        const topConfidence = result.data[0].confidences[0].confidence;

        results.innerHTML = `
            <h3>🧘 Predicted Pose</h3>
            <p style="font-size:22px; font-weight:bold;">
                ${topLabel}
            </p>
            <p>
                📊 Confidence: ${(topConfidence * 100).toFixed(2)}%
            </p>
        `;

    } catch (error) {
        console.error(error);
        results.innerHTML = "❌ Prediction failed. Check console.";
    }
});