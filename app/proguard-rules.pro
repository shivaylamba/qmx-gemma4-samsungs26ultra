# Native methods are invoked through JNI by the inference library.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}
