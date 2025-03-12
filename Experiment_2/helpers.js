function saveData() {
    var completion_date = new Date();
    jsPsych.data.get().push({ success: true, completion_time: completion_date, exp_type: "gesture-beat-expt-pilot" });
  
    var full_data = jsPsych.data.get().json();
    console.log(full_data);
    $.ajax({
        type: "POST",
        url: "write_data.php",
        data: JSON.stringify({ 'prolific_data': jsPsych.data.dataProperties, 'data': full_data, 'dir_path': "data" }),
        contentType: "application/json"
    })
        .done(function () {
            window.location.href = "finish.html";
            // alert('Successfully submit data.')
            // console.log(full_data);
        })
        .fail(function () {
            alert("A problem occurred while writing to the database. Please contact the researcher for more information.")
            // window.location.href = "/";
        })
  }

function showSeekWarning() {
    document.getElementById("seeking-alert").style.display = "block";
}

function disableSeeking(v) {
    // Disable seeking
    var supposedCurrentTime = 0;
    v.addEventListener('timeupdate', function() {
    if (!v.seeking) {
        supposedCurrentTime = v.currentTime;
    }
    });
    // prevent user from seeking
    v.addEventListener('seeking', function() {
        // guard agains infinite recursion:
        // user seeks, seeking is fired, currentTime is modified, seeking is fired, current time is modified, ....
        var delta = v.currentTime - supposedCurrentTime;
        if (Math.abs(delta) > 0.01) {
            console.log("Seeking is disabled");
            v.currentTime = supposedCurrentTime;
        }
        showSeekWarning();

    });
    // delete the following event handler if rewind is not required
    v.addEventListener('ended', function() {
    // reset state in order to allow for rewind
        supposedCurrentTime = 0;
    });
}