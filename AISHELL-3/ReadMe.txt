
		       	            AISHELL-3

			        北京希尔贝壳科技有限公司
      	       Beijing Shell Shell Technology Co.,Ltd
			              10/18/2020

1. AISHELL-3 Speech Data

		- Sampling Rate : 	44.1kHz
		- Sample Format : 	16bit
		- Environment :		Quiet indoor
		- Speech Data Type : 	PCM
		- Channel Number : 	1
		- Recording Equipment : 	High fidelity microphone
		- Sentences : 	88035 utterances
		- Speaker : 	218 speakers (43 male and 175 female)


2. Data Structure
│  README.txt			                                        （readme）
│  ChangeLog			                                        （Change Information）
│  phone_set.txt			                                    （phone Information）
│  spk_info.txt                                                 （Speaker Information）
└─ test				                          	                （Test Data File）
└─ train				                          	            （Train Data File）
	│├─content.txt                                              （Transcript Content）
	│├─prosody_label_train-set.txt                              （Prosody Lable）
	│├─wav                                                      （Audio Data File）
	   │├─SSB005						                        （Speaker ID File）
       ││  ││      ID2166W0001.wav			                    （Audio）

4. System
		AISHELL-3 is a large-scale and high-fidelity multi-speaker Mandarin speech corpus which could be used to                                 train multi-speaker Text-to-Speech (TTS) systems.
		You can download data set from: http://www.aishelltech.com/aishell_3.
		The baseline system code and generated samples are available online form:                                 https://sos1sos2sixteen.github.io/aishell3/.
