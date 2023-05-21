function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        let cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            let cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

let csrftoken = getCookie('csrftoken');

$.ajaxSetup({
    beforeSend: function (xhr, settings) {
        if (!this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});

$(".delete-image").click(function () {
    var imageId = $(this).data("id");
    console.log("Image ID: ", imageId);  // Add this line

    $.ajax({
        url: '/delete_image/' + imageId + '/',
        type: 'POST',
        data: {
            'csrfmiddlewaretoken': $('input[name="csrfmiddlewaretoken"]').val()
        },
        success: function (response) {
            if (response.success) {
                $('#image-' + imageId).remove();
            }
        }
    });
});

document.getElementById('upload-button').addEventListener('click', function () {
    var fileInput = document.getElementById('file-input');
    fileInput.click(); // Trigger the file input click event
});

document.getElementById('file-input').addEventListener('change', function () {
    var fileInput = document.getElementById('file-input');
    var file = fileInput.files[0]; // Get the selected file

    // Create a new FormData instance
    var formData = new FormData();
    formData.append('image', file); // Append the selected file to the form data

    // Display loading overlay
    var loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'block';
    }

    // Disable upload button
    var uploadButton = document.getElementById('upload-button');
    if (uploadButton) {
        uploadButton.disabled = true;
    }

    fetch('/home/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCookie('csrftoken'),
        },
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Image successfully uploaded and art generated
                // Perform any necessary actions (e.g., display the generated art)
                console.log('Art generated successfully');
                // Hide loading overlay
                if (loadingOverlay) {
                    loadingOverlay.style.display = 'none';
                }
                // Enable upload button
                if (uploadButton) {
                    uploadButton.disabled = false;
                }
                // Reload the page to show the generated art
                location.reload();
            } else {
                // Error occurred during upload or art generation
                // Handle the error appropriately
                console.log('Error occurred');
                // Hide loading overlay
                if (loadingOverlay) {
                    loadingOverlay.style.display = 'none';
                }
                // Enable upload button
                if (uploadButton) {
                    uploadButton.disabled = false;
                }
            }
        })
        .catch(error => {
            // Error occurred during the fetch request
            // Handle the error appropriately
            console.log('Error occurred');
            // Hide loading overlay
            if (loadingOverlay) {
                loadingOverlay.style.display = 'none';
            }
            // Enable upload button
            if (uploadButton) {
                uploadButton.disabled = false;
            }
        });
});
