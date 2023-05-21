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

document.addEventListener('DOMContentLoaded', function () {
    // Select the loading overlay
    var loadingOverlay = document.getElementById('loading-overlay');

    // Select the file input element
    var fileInput = document.getElementById('file-input');

    // Select the upload button
    var uploadButton = document.getElementById('upload-button');

    // Add event listener to the upload button
    uploadButton.addEventListener('click', function () {
        // Check if a file is selected
        if (fileInput.files.length > 0) {
            // Show the loading overlay
            loadingOverlay.style.display = 'block';

            // Disable the upload button
            uploadButton.disabled = true;

            // Create a new FormData instance
            var formData = new FormData();
            formData.append('image', fileInput.files[0]); // Append the selected file to the form data

            // Perform the file upload using AJAX
            $.ajax({
                url: '/home/',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function (response) {
                    if (response.success) {
                        // Image successfully uploaded and art generated
                        // Perform any necessary actions (e.g., display the generated art)
                        console.log('Art generated successfully');
                        // Hide the loading overlay
                        loadingOverlay.style.display = 'none';
                        // Reload the page to show the generated art
                        location.reload();
                    } else {
                        // Error occurred during upload or art generation
                        // Handle the error appropriately
                        console.log('Error occurred');
                        // Hide the loading overlay
                        loadingOverlay.style.display = 'none';
                    }
                    // Enable the upload button
                    uploadButton.disabled = false;
                },
                error: function () {
                    // Error occurred during the AJAX request
                    // Handle the error appropriately
                    console.log('Error occurred');
                    // Hide the loading overlay
                    loadingOverlay.style.display = 'none';
                    // Enable the upload button
                    uploadButton.disabled = false;
                }
            });
        }
    });

    // Hide the loading overlay initially
    loadingOverlay.style.display = 'none';
});
