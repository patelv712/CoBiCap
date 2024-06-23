function uploadImage() {
    var fileIn = document.getElementById('imageInput');
    var f = fileIn.files[0];
    var data = new FormData();
    data.append('image', f);
  
    fetch('/upload', {
      method: 'POST',
      body: data
    })
    .then(response => 
      response.json())
    .then(data => {
      document.getElementById('caption').innerText = '<b>Caption</b>: ' + data.caption;
    })
    .catch(error => 
      console.error('Error:', error));
  }
  