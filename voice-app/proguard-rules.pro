# Native method names are resolved through JNI.
-keepclasseswithmembernames class * {
    native <methods>;
}
