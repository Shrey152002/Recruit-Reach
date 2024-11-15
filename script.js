document.getElementById("linkForm").addEventListener("submit", function (e) {
    e.preventDefault(); // Prevents the page from refreshing on form submission

    const formData = new FormData();
    formData.append("link", document.getElementById("linkInput").value);

    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = "Processing...";

    fetch("/process", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                outputDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                outputDiv.innerHTML = `<pre>${data.content}</pre>`;
            }
        })
        .catch((error) => {
            outputDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        });
});
