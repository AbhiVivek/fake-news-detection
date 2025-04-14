async function checkFakeNews() {
    const text = document.getElementById("newsText").value.trim();
    const resultDiv = document.getElementById("result");

    // Check if input text is provided
    if (!text) {
        resultDiv.innerHTML = "Please enter the news text.";
        resultDiv.className = 'result';
        resultDiv.classList.add('fake');
        return;
    }

    try {
        // Call backend API to check news
        const response = await fetch("https://fake-news-detection-romh.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (response.ok) {
            // Display result
            const result = data.prediction;
            resultDiv.innerHTML = result === "Fake News" ? "This is Fake News!" : "This is Real News!";
            resultDiv.className = 'result';
            resultDiv.classList.add(result.toLowerCase().replace(" ", "-"));
        } else {
            resultDiv.innerHTML = data.error || "Something went wrong.";
            resultDiv.className = 'result';
            resultDiv.classList.add('fake');
        }
    } catch (error) {
        resultDiv.innerHTML = "Error connecting to the backend. Please try again later.";
        resultDiv.className = 'result';
        resultDiv.classList.add('fake');
    }
}
