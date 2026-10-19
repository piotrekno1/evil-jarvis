#import <Foundation/Foundation.h>
#import <Speech/Speech.h>

static void runLoopUntil(BOOL *done, NSTimeInterval timeout) {
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
    while (!*done && [deadline timeIntervalSinceNow] > 0) {
        [[NSRunLoop currentRunLoop]
            runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
    }
}

static int finish(NSString *outputPath, NSString *text, NSString *error, int code) {
    NSDictionary *payload = @{
        @"text": text ?: @"",
        @"error": error ?: @"",
        @"code": @(code),
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:nil];
    if (![data writeToFile:outputPath atomically:YES]) {
        return 70;
    }
    return code;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr, "usage: AppleSpeechHelper AUDIO.wav LOCALE OUTPUT.json\n");
            return 64;
        }

        NSString *audioPath = [NSString stringWithUTF8String:argv[1]];
        NSString *localeID = [NSString stringWithUTF8String:argv[2]];
        NSString *outputPath = [NSString stringWithUTF8String:argv[3]];
        SFSpeechRecognizer *recognizer = [[SFSpeechRecognizer alloc]
            initWithLocale:[[NSLocale alloc] initWithLocaleIdentifier:localeID]];
        if (recognizer == nil) {
            return finish(outputPath, nil,
                          [NSString stringWithFormat:@"Apple Speech does not support locale %@", localeID],
                          2);
        }

        __block BOOL authorizationDone = NO;
        __block SFSpeechRecognizerAuthorizationStatus authorizationStatus =
            [SFSpeechRecognizer authorizationStatus];
        if (authorizationStatus == SFSpeechRecognizerAuthorizationStatusNotDetermined) {
            [SFSpeechRecognizer requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus status) {
                authorizationStatus = status;
                authorizationDone = YES;
            }];
            runLoopUntil(&authorizationDone, 30.0);
            if (!authorizationDone) {
                return finish(outputPath, nil,
                              @"Timed out waiting for Speech Recognition permission", 3);
            }
        }
        if (authorizationStatus != SFSpeechRecognizerAuthorizationStatusAuthorized) {
            return finish(outputPath, nil,
                          [NSString stringWithFormat:
                              @"Speech Recognition permission was not granted (status %ld)",
                              (long)authorizationStatus],
                          4);
        }
        if (![recognizer supportsOnDeviceRecognition]) {
            return finish(outputPath, nil,
                          [NSString stringWithFormat:
                              @"On-device recognition is unavailable for locale %@", localeID],
                          5);
        }

        NSURL *audioURL = [NSURL fileURLWithPath:audioPath];
        SFSpeechURLRecognitionRequest *request =
            [[SFSpeechURLRecognitionRequest alloc] initWithURL:audioURL];
        request.requiresOnDeviceRecognition = YES;
        request.shouldReportPartialResults = NO;

        __block BOOL recognitionDone = NO;
        __block NSString *transcript = nil;
        __block NSError *recognitionError = nil;
        SFSpeechRecognitionTask *task = [recognizer
            recognitionTaskWithRequest:request
            resultHandler:^(SFSpeechRecognitionResult *result, NSError *error) {
                if (result != nil) {
                    transcript = result.bestTranscription.formattedString;
                    if (result.isFinal) {
                        recognitionDone = YES;
                    }
                }
                if (error != nil) {
                    recognitionError = error;
                    recognitionDone = YES;
                }
            }];
        runLoopUntil(&recognitionDone, 20.0);
        if (!recognitionDone) {
            [task cancel];
            return finish(outputPath, nil, @"Apple Speech transcription timed out", 6);
        }
        if (transcript.length == 0 && recognitionError != nil) {
            return finish(outputPath, nil,
                          [NSString stringWithFormat:@"Apple Speech failed: %@",
                              recognitionError.localizedDescription],
                          7);
        }
        return finish(outputPath, transcript, nil, 0);
    }
}
