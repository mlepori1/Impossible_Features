/**************************************************************************
 * INITIALIZATION
**************************************************************************/

// Initialize jsPsych.
var jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,
  on_finish: function() {
    jsPsych.data.displayData();
    // saveData();
  }
});

// Capture info from Prolific.
var subject_id = jsPsych.data.getURLVariable('PROLIFIC_PID');
var study_id = jsPsych.data.getURLVariable('STUDY_ID');
var session_id = jsPsych.data.getURLVariable('SESSION_ID');
jsPsych.data.addProperties({
    subject_id: subject_id,
    study_id: study_id,
    session_id: session_id
});

// Set random seed.
const seed = jsPsych.randomization.setSeed();
jsPsych.data.addProperties({
    rng_seed: seed
});

/**************************************************************************
 * GLOBAL VARIABLES 
**************************************************************************/

var DEBUG = true; // CHANGE TO FALSE FOR REAL EXPERIMENT
var USE_COULD = true; // whether or not to use could vs can
var REQUIRE_QUESTIONS = !DEBUG; 

var STIM_COUNT = 20;
var TRAIN_COUNT = 4;
var ATTN_COUNT = 4;


// Get stimuli according to list ID.
var stimuli = test_stimuli; // test_stimuli is read from prefixes_stimuli.js
if (DEBUG) {
  STIM_COUNT = 4; // just use a small number of stimuli when debugging
}

stimuli = jsPsych.randomization.sampleWithoutReplacement(stimuli, STIM_COUNT + TRAIN_COUNT); // Sample stimuli randomly
var train_stimuli = stimuli.slice(0, TRAIN_COUNT)
var test_stimuli = stimuli.slice(TRAIN_COUNT, TRAIN_COUNT + STIM_COUNT)

// Repeat test stimuli three times
test_stimuli = test_stimuli.concat(test_stimuli, test_stimuli);
var n_trials = test_stimuli.length + ATTN_COUNT; 

// For deciding when to deploy attention checks
var n_exp_trials = test_stimuli.length;
var exp_trial_progress = 0.0;
var ATTN_INCREMENT = 1 / ATTN_COUNT;
var ATTN_THRESHOLD = ATTN_INCREMENT;

// Conditions.
var CONDITIONS = [
  "probable", "improbable", "impossible", "inconceivable"
];

const PROMPT_TYPE_MAP = new Map() 
PROMPT_TYPE_MAP.set("improbable", "improbable")
PROMPT_TYPE_MAP.set("impossible", "impossible")
PROMPT_TYPE_MAP.set("inconceivable", "nonsensical")

var PROMPT_TYPE = ["improbable", "impossible", "inconceivable"]


// These variables will be updated on each trial.
var CUR_CONDITION = "";
var CUR_PROMPT = "";
var CUR_QUERY = "";

/**************************************************************************
 * HELPER FUNCTIONS
**************************************************************************/



function get_stimulus(verb, object, prep, continuation, condition) {
  // if (USE_COULD) {
  //   prefix = prefix.replace("Can", "Could")
  // };
  // var s = prefix + " " + continuation + "?"
  var task = "Rate the following phrase according to how <b>" + condition + "</b> it is:<BR/><BR/>\""
  var s = task + verb + " " + object + " " + prep + " " + continuation + "\""
  s = s.replace("[POSS]", "their")
  return s
}

function get_s_condition_order(COUNT) {
  // There are COUNT test/train stimuli in total, and 4 stim conditions, so make sure everyone
  // sees COUNT/4 examples of each condition
  var s = [];
  CONDITIONS.forEach((cond) => s = s.concat(Array(COUNT/4).fill(cond)))
  return s
}

function get_q_condition_order(COUNT) {
  // Generate a list of COUNT instances of each q_condition
  var q = [];
  PROMPT_TYPE.forEach((cond) => q = q.concat(Array(COUNT).fill(cond)))
  return q
}

function shuffle_together(l1, l2, l3) {
  // There are STIM_COUNT * 3 stimuli
  var index = Array.from(Array(l1.length).keys())
  index = jsPsych.randomization.shuffle(index)

  l1 = index.map(i => l1[i]);
  l2 = index.map(i => l2[i]);
  l3 = index.map(i => l3[i]);

  return [l1, l2, l3];
}
// Generate train stimulus condition order and 
// test stimulus condition order repeated three times
var S_COND_ORDER_TRAIN= get_s_condition_order(TRAIN_COUNT);

var S_COND_ORDER = get_s_condition_order(STIM_COUNT);
S_COND_ORDER = S_COND_ORDER.concat(S_COND_ORDER, S_COND_ORDER);

// Generate question condition order for train and test
var Q_COND_ORDER_TRAIN = get_q_condition_order(1);
Q_COND_ORDER_TRAIN = Q_COND_ORDER_TRAIN.concat(Q_COND_ORDER_TRAIN).slice(0, TRAIN_COUNT)

var Q_COND_ORDER = get_q_condition_order(STIM_COUNT);

// Shuffle Test Stimuli together
[test_stimuli, S_COND_ORDER, Q_COND_ORDER] = shuffle_together(test_stimuli, S_COND_ORDER, Q_COND_ORDER);

/**************************************************************************
 * EXPERIMENT CODE
**************************************************************************/

/* create timeline */
var timeline = [];

/* define instructions trial */
var instructions = {
  type: jsPsychInstructions,
  pages: [
    `<div class="jspsych-content" align=left style="width:1000px;text-align: left;">
    <h2>Hello, and welcome to our study!</h2>
    This study will consist of two short phases. First, we will ask you ${TRAIN_COUNT} simple questions to familiarize
    you with the task. Next, we will ask you ${n_trials} similar questions.
    <br><br>
    The questions are all about rating how improbable, impossible, or nonsensical particular scenarios are.
    <br>
    <h3>Your task:</h3>
    <ul>
      <li>Read each question carefully.</li>
      <li>Answer how improbable, impossible, or nonsensical the sentence is.</li>
      <li>You will answer using a slider.</li>
      <li>Respond as quickly as you can.</li>
    </ul>
    <h3>IMPORTANT:</h3>
    <ul>
      <li>Answer the questions based on what is possible given the physical laws of the real world.</li>
      <li>Some questions might sound weird or nonsensical. Just answer the best you can.</li>
      <li>There are no right or wrong answers. We are simply interested in your intuitions.</li>
      <li>Remember to respond quickly!</li>
    </ul>
    <br>
    Once you are ready, click the button below to begin the first phase of the study.
    </div>
    <br>`
  ],
  button_label_next: "Begin Phase 1",
  allow_backward: false,
  show_clickable_nav: true
}
instructions.on_finish = function (data) {data.task_type = "instructions";};
timeline.push(instructions);

// Define the behavior of the training trials, which is identical to test trails
// except using different stimuli
var train_trial = {
  type: jsPsychHtmlSliderResponse,
  data: {},
  stimulus: function() {
    // Get last element of pre-generated order of conditions.
    CUR_CONDITION = S_COND_ORDER_TRAIN.shift();
    CUR_PROMPT = Q_COND_ORDER_TRAIN.shift();

    CUR_QUERY = get_stimulus(
      jsPsych.timelineVariable("verb_participle"), 
      jsPsych.timelineVariable("object"),
      jsPsych.timelineVariable("prep"),
      jsPsych.timelineVariable(CUR_CONDITION),
      PROMPT_TYPE_MAP.get(CUR_PROMPT),

    );
    var html = `<div style="font-size:20px;"><p>${CUR_QUERY}</p></div>`;
    return html
  },

  require_movement: true,
  min: 0,
  max: 100,
  // step: 1,
  labels: function() {
    return ["Normal", `More ${PROMPT_TYPE_MAP.get(CUR_PROMPT)} &#8594`, ""]
  },
  slider_start: 50,
  slider_width: 400
};
train_trial.on_start = function(train_trial){
  train_trial.data = jsPsych.getAllTimelineVariables();
  train_trial.data.task_type = "training";
};
train_trial.on_finish = function(data){
  // at the end of each trial, update the progress bar
  // based on the current value and the proportion to update for each trial
  var cur_progress_bar_value = jsPsych.getProgressBarCompleted();
  jsPsych.setProgressBar(cur_progress_bar_value + (1/TRAIN_COUNT));

  // Save other variables.
  data.condition = CUR_CONDITION;
  data.prompt = CUR_PROMPT
  data.query = CUR_QUERY;
};


/* define train_trial procedure */
var train_procedure = {
  timeline: [train_trial],
  timeline_variables: train_stimuli,
  randomize_order: false
};
timeline.push(train_procedure);

/* define ready trial, which occurs between train and test */
var ready = {
  type: jsPsychInstructions,
  pages: [
    `<div class="jspsych-content" align=left style="width:1000px;text-align: left;">
    We will now commence the main study. We will ask you ${n_trials} simple questions. <b>Every once in a while, we will ask you a question to ensure that you are paying attention to the study!</b>
    <br>
    <h3>REMINDER:</h3>
    <br>
    The questions are all about rating how improbable, impossible, or nonsensical particular scenarios are.
    <br>
    <h3>Your task:</h3>
    <ul>
      <li>Read each question carefully.</li>
      <li>Answer how improbable, impossible, or nonsensical the sentence is, unless instructed to do otherwise.</li>
      <li>You will answer most questions using a slider. Answer the remaining questions by clicking on your choice.</li>
      <li>Respond as quickly as you can.</li>
    </ul>
    <h3>IMPORTANT:</h3>
    <ul>
      <li>Answer the questions based on what is possible given the physical laws of the real world.</li>
      <li>Some questions might sound weird or nonsensical. Just answer the best you can.</li>
      <li>There are no right or wrong answers. We are simply interested in your intuitions.</li>
      <li>Remember to respond quickly!</li>
    </ul>
    <br>
    Once you are ready, click the button below to begin the second phase of the study.
    </div>
    <br>`
  ],
  button_label_next: "Begin Phase 2",
  allow_backward: false,
  show_clickable_nav: true
}
ready.on_finish = function (data) {
  data.task_type = "ready";
  cur_progress_bar_value = 0;
  jsPsych.setProgressBar(0);

};
timeline.push(ready);


// define behavior of main experimental trails
var trial = {
  type: jsPsychHtmlSliderResponse,
  data: {},
  stimulus: function() {
    // Get last element of pre-generated order of conditions.
    CUR_CONDITION = S_COND_ORDER.shift();
    CUR_PROMPT = Q_COND_ORDER.shift();

    CUR_QUERY = get_stimulus(
      jsPsych.timelineVariable("verb_participle"), 
      jsPsych.timelineVariable("object"),
      jsPsych.timelineVariable("prep"),
      jsPsych.timelineVariable(CUR_CONDITION),
      PROMPT_TYPE_MAP.get(CUR_PROMPT),

    );
    var html = `<div style="font-size:20px;"><p>${CUR_QUERY}</p></div>`;
    return html
  },

  require_movement: true,
  min: 0,
  max: 100,
  // step: 1,
  labels: function() {
    return ["Normal", `More ${PROMPT_TYPE_MAP.get(CUR_PROMPT)} &#8594`, ""]
  },
  slider_start: 50,
  slider_width: 400
};
trial.on_start = function(trial){
  trial.data = jsPsych.getAllTimelineVariables();
  trial.data.task_type = "critical";
};
trial.on_finish = function(data){
  // at the end of each trial, update the progress bar
  // based on the current value and the proportion to update for each trial
  var cur_progress_bar_value = jsPsych.getProgressBarCompleted();
  jsPsych.setProgressBar(cur_progress_bar_value + (1/n_trials));

  exp_trial_progress = exp_trial_progress + (1/n_exp_trials);
  console.log(exp_trial_progress)
  console.log(ATTN_THRESHOLD);

  // Save other variables.
  // data.counterbalance = CUR_COUNTERBALANCE;
  data.condition = CUR_CONDITION;
  data.prompt = CUR_PROMPT
  data.query = CUR_QUERY;
};


var optional_attention_check = {
  type: jsPsychHtmlButtonResponse,
  data: {},
  stimulus: `<div style="font-size:20px;"><p>What did the previous question ask you to do?</p></div>`,
  choices: [
    "Rate the phrase according to how <b>improbable</b> it is",
    "Rate the phrase according to how <b>impossible</b> it is",
    "Rate the phrase according to how <b>nonsensical</b> it is",
  ]
};
optional_attention_check.on_start = function(optional_attention_check){
  optional_attention_check.data = jsPsych.getAllTimelineVariables();
  optional_attention_check.data.task_type = "attention_check";
  optional_attention_check.data.correct_answer = CUR_QUERY;
  optional_attention_check.data.choices = [
    "Rate the phrase according to how <b>improbable</b> it is",
    "Rate the phrase according to how <b>impossible</b> it is",
    "Rate the phrase according to how <b>nonsensical</b> it is",
  ];


};
optional_attention_check.on_finish = function(data){
  // at the end of each trial, update the progress bar
  // based on the current value and the proportion to update for each trial
  var cur_progress_bar_value = jsPsych.getProgressBarCompleted();
  jsPsych.setProgressBar(cur_progress_bar_value + (1/n_trials));
};


var attention_bool = {
  timeline: [optional_attention_check],
  conditional_function: function(){
      if(exp_trial_progress >= ATTN_THRESHOLD){
        ATTN_THRESHOLD = ATTN_THRESHOLD + ATTN_INCREMENT;
        console.log(ATTN_THRESHOLD);
        return true;
      } else {
        return false;
      }
  }
}


/* define test procedure */
var test_procedure = {
  timeline: [trial, attention_bool],
  timeline_variables: test_stimuli,
  randomize_order: false
};
timeline.push(test_procedure);

/* define postquestionnaire */
var post_test_survey = {
  type: jsPsychSurveyText,
  preamble: `
    <p>Thanks for contributing to our study, and have a nice day!</p>
    <p>The following questions are optional, but your feedback would really help us improve our study for future participants.</p>
  `,
  questions: [
    {prompt: "What is your native language?", name: 'language'},
    {prompt: "What is your gender?", name: 'gender'},
    {prompt: "Please leave any additional feedback in the text box below.", name: 'feedback', rows: 10}
  ],
  on_finish: function(data) {data.task_type = "survey"}
};
timeline.push(post_test_survey);

/* start the experiment */
function run_experiment() {
    jsPsych.run(timeline);
}