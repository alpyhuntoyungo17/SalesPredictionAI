document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const predictionResult = document.getElementById('predictionResult');

    predictionForm.addEventListener('submit', async function(event) {
        event.preventDefault();

        const product = document.getElementById('product').value;
        const units = document.getElementById('units').value;

        const predictionData = {
            product: product,
            units: parseInt(units)
        };

        try {
            const prediction = await predictSales(predictionData);
            displayPrediction(prediction);
        } catch (error) {
            console.error('Prediction request failed:', error);
            displayErrorMessage('Failed to predict sales. Please try again later.');
        }
    });

    async function predictSales(data) {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error('Failed to predict sales.');
        }

        return await response.json();
    }

    function displayPrediction(prediction) {
        predictionResult.innerHTML = `
            <h3>Prediction Result</h3>
            <p><strong>Product:</strong> ${prediction.product}</p>
            <p><strong>Units Sold:</strong> ${prediction.units}</p>
            <p><strong>Predicted Revenue:</strong> $${prediction.revenue.toFixed(2)}</p>
        `;
    }

    function displayErrorMessage(message) {
        predictionResult.innerHTML = `<p class="error">${message}</p>`;
    }
});
