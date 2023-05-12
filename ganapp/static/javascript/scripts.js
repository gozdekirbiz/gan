$(".delete-image").click(function () {
    var imageId = $(this).data('id');
    $.ajax({
        url: '/delete_image/' + imageId,
        type: 'DELETE',
        success: function (result) {
            location.reload();
        }
    });
});
