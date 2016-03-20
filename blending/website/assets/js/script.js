$(document).ready(function(){
    $('.photo').click(function() {
        var src = $(this).attr('src');
        $('#photo-modal').attr('src',src);
        showModal();
    });
    var mouseInsideModal = false;
    $('#photo-modal').hover(function(){ 
        mouseInsideModal=true; 
    }, function(){ 
        mouseInsideModal=false; 
    });
    
    $('#modal-content').click(function() {
        if (!mouseInsideModal) {
            hideModal();   
        }
    });
    
    function showModal(){
        $("#modal").fadeIn();
    }

    function hideModal(){
        $("#modal").fadeOut();
    }
});