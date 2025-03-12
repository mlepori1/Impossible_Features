<?php
// get the data from the POST message
$post_data = json_decode(file_get_contents('php://input'), true);
$data = $post_data['data'];
$dir_path = $post_data['dir_path'];
// generate a unique ID for the file, e.g., session-6feu833950202 
$file = uniqid("session-");
// the directory "data" must be writable by the server
$name = "{$dir_path}/{$file}.json";
// write the file to disk
file_put_contents($name, $data);
?>
