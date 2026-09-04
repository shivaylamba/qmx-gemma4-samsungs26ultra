#include <jni.h>

#include <cstdint>
#include <memory>
#include <string>

#include "phonemizer.h"

namespace {

void throw_java(JNIEnv * env, const char * type, const std::string & message) {
    if (const jclass clazz = env->FindClass(type); clazz != nullptr) {
        env->ThrowNew(clazz, message.c_str());
    }
}

class UtfChars {
public:
    UtfChars(JNIEnv * env, jstring value) : env_(env), value_(value) {
        chars_ = value == nullptr ? nullptr : env->GetStringUTFChars(value, nullptr);
    }

    ~UtfChars() {
        if (chars_ != nullptr) env_->ReleaseStringUTFChars(value_, chars_);
    }

    const char * get() const { return chars_; }

private:
    JNIEnv * env_;
    jstring value_;
    const char * chars_ = nullptr;
};

IPAPhonemizer * from_handle(jlong handle) {
    return reinterpret_cast<IPAPhonemizer *>(static_cast<intptr_t>(handle));
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_qmxassistant_KittenPhonemizer_nativeCreate(
        JNIEnv * env,
        jobject,
        jstring rules_path,
        jstring list_path) {
    UtfChars rules(env, rules_path);
    UtfChars list(env, list_path);
    if (rules.get() == nullptr || list.get() == nullptr) {
        throw_java(env, "java/lang/IllegalArgumentException", "Phonemizer dictionary paths are required");
        return 0;
    }

    try {
        auto phonemizer = std::make_unique<IPAPhonemizer>(rules.get(), list.get(), "en-us");
        if (!phonemizer->isLoaded()) {
            const std::string error = phonemizer->getError();
            throw_java(env, "java/lang/IllegalStateException", error.empty() ? "Could not load phonemizer data" : error);
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<intptr_t>(phonemizer.release()));
    } catch (const std::exception & error) {
        throw_java(env, "java/lang/IllegalStateException", error.what());
    } catch (...) {
        throw_java(env, "java/lang/IllegalStateException", "Could not initialize the KittenTTS phonemizer");
    }
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_qmxassistant_KittenPhonemizer_nativePhonemize(
        JNIEnv * env,
        jobject,
        jlong handle,
        jstring text) {
    auto * phonemizer = from_handle(handle);
    if (phonemizer == nullptr) {
        throw_java(env, "java/lang/IllegalStateException", "KittenTTS phonemizer is not loaded");
        return nullptr;
    }
    UtfChars input(env, text);
    if (input.get() == nullptr) return nullptr;

    try {
        const std::string result = phonemizer->phonemizeText(input.get());
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception & error) {
        throw_java(env, "java/lang/IllegalStateException", error.what());
    } catch (...) {
        throw_java(env, "java/lang/IllegalStateException", "KittenTTS phonemization failed");
    }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_qmxassistant_KittenPhonemizer_nativeDestroy(
        JNIEnv *,
        jobject,
        jlong handle) {
    delete from_handle(handle);
}
