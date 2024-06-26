document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const result = document.getElementById('result');
    const predictionValue = document.getElementById('prediction-value');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = {
            day_of_week: document.getElementById('day_of_week').value,
            month: parseInt(document.getElementById('month').value),
            category: document.getElementById('category').value,
            store_id: document.getElementById('store_id').value,
            weather_condition: document.getElementById('weather_condition').value,
            avg_price: parseFloat(document.getElementById('avg_price').value),
            is_weekend: parseInt(document.getElementById('is_weekend').value),
            is_holiday: parseInt(document.getElementById('is_holiday').value),
            promotion_active: parseInt(document.getElementById('promotion_active').value),
            temperature: parseFloat(document.getElementById('temperature').value)
        };

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                predictionValue.textContent = `$${data.prediction.toFixed(2)}`;
                result.classList.remove('hidden');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    });
});
