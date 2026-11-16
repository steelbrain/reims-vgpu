//! The known keys: the attachment and extension keys a sample buffer serialization writes as a
//! 16-bit index (`idxk`) rather than as a string (`strk`).
//!
//! The table's order is CoreMedia's serializer's, which names each entry by its constant; the
//! strings are those public CoreMedia and CoreVideo constants' values, which do not change between
//! releases, and the entries it spells as literals are those literals.
//!
//! Some strings appear more than once (`Width` at 46 and 55, `Height` at 47, 56 and 148, among
//! others). A reader maps every index to its string.

/// The table, index order.
pub const KNOWN_KEYS: [&str; 212] = [
    "NotSync",                        // kCMSampleAttachmentKey_NotSync
    "PartialSync",                    // kCMSampleAttachmentKey_PartialSync
    "HasRedundantCoding",             // kCMSampleAttachmentKey_HasRedundantCoding
    "IsDependedOnByOthers",           // kCMSampleAttachmentKey_IsDependedOnByOthers
    "DependsOnOthers",                // kCMSampleAttachmentKey_DependsOnOthers
    "EarlierDisplayTimesAllowed",     // kCMSampleAttachmentKey_EarlierDisplayTimesAllowed
    "DisplayImmediately",             // kCMSampleAttachmentKey_DisplayImmediately
    "DoNotDisplay",                   // kCMSampleAttachmentKey_DoNotDisplay
    "BytesOfClearDataCount",          // kCMSampleAttachmentKey_BytesOfClearDataCount
    "CryptorIV",                      // kCMSampleAttachmentKey_CryptorInitializationVector
    "ResetDecoderBeforeDecoding",     // kCMSampleBufferAttachmentKey_ResetDecoderBeforeDecoding
    "DrainAfterDecoding",             // kCMSampleBufferAttachmentKey_DrainAfterDecoding
    "PostNotificationWhenConsumed",   // kCMSampleBufferAttachmentKey_PostNotificationWhenConsumed
    "ResumeOutput",                   // kCMSampleBufferAttachmentKey_ResumeOutput
    "TrimDurationAtStart",            // kCMSampleBufferAttachmentKey_TrimDurationAtStart
    "TrimDurationAtEnd",              // kCMSampleBufferAttachmentKey_TrimDurationAtEnd
    "SpeedMultiplier",                // kCMSampleBufferAttachmentKey_SpeedMultiplier
    "Reverse",                        // kCMSampleBufferAttachmentKey_Reverse
    "FillDiscontinuitiesWithSilence", // kCMSampleBufferAttachmentKey_FillDiscontinuitiesWithSilence
    "EmptyMedia",                     // kCMSampleBufferAttachmentKey_EmptyMedia
    "PermanentEmptyMedia",            // kCMSampleBufferAttachmentKey_PermanentEmptyMedia
    "DisplayEmptyMediaImmediately",   // kCMSampleBufferAttachmentKey_DisplayEmptyMediaImmediately
    "EndsPreviousSampleDuration",     // kCMSampleBufferAttachmentKey_EndsPreviousSampleDuration
    "SampleReferenceURL",             // kCMSampleBufferAttachmentKey_SampleReferenceURL
    "SampleReferenceByteOffset",      // kCMSampleBufferAttachmentKey_SampleReferenceByteOffset
    "GradualDecoderRefresh",          // kCMSampleBufferAttachmentKey_GradualDecoderRefresh
    "SourceRect",                     // kFigSampleBufferAttachmentKey_SourceRect
    "DestRect",                       // kFigSampleBufferAttachmentKey_DestRect
    "VirtualDisplaySizeForDestRect",  // kFigSampleBufferAttachmentKey_VirtualDisplaySizeForDestRect
    "SymmetricTransform",             // kFigSampleBufferAttachmentKey_SymmetricTransform
    "LoudnessInfo",                   // kFigSampleBufferAttachmentKey_LoudnessInfo
    "TransitionID",                   // kCMSampleBufferAttachmentKey_TransitionID
    "StartPresentationTimesStamp", // kCMSampleBufferConduitNotificationParameter_StartPresentationTimeStamp
    "EndPresentationTimesStamp", // kCMSampleBufferConduitNotificationParameter_EndPresentationTimeStamp
    "EditBoundary",              // kCMSampleBufferAttachmentKey_EditBoundary
    "EmptyMediaBeforeBeginning", // kCMSampleBufferAttachmentKey_EmptyMediaBeforeBeginning
    "ClientTimingInfo",          // kCMSampleBufferAttachmentKey_ClientTimingInfo
    "ReferenceWasRefreshed",     // kVTSampleAttachmentKey_ReferenceWasRefreshed
    "EncoderRetryCount",         // kVTSampleAttachmentKey_EncoderRetryCount
    "RequireAcknowledgementToken", // kVTSampleAttachmentKey_RequireAcknowledgementToken
    "value",                     // kCMTimeValueKey
    "timescale",                 // kCMTimeScaleKey
    "epoch",                     // kCMTimeEpochKey
    "flags",                     // kCMTimeFlagsKey
    "X",
    "Y",
    "Width",
    "Height",
    "com.apple.fig.format_description.original_compression_settings", // kCMFormatDescriptionExtension_OriginalCompressionSettings
    "SampleDescriptionExtensionAtoms", // kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms
    "VerbatimSampleDescription",       // kCMFormatDescriptionExtension_VerbatimSampleDescription
    "VerbatimISOSampleEntry",          // kCMFormatDescriptionExtension_VerbatimISOSampleEntry
    "FormatName",                      // kCMFormatDescriptionExtension_FormatName
    "Depth",                           // kCMFormatDescriptionExtension_Depth
    "CVCleanAperture",                 // kCVImageBufferCleanApertureKey
    "Width",                           // kCVImageBufferCleanApertureWidthKey
    "Height",                          // kCVImageBufferCleanApertureHeightKey
    "HorizontalOffset",                // kCVImageBufferCleanApertureHorizontalOffsetKey
    "VerticalOffset",                  // kCVImageBufferCleanApertureVerticalOffsetKey
    "HeightAsRational",                // kCMFormatDescriptionKey_CleanApertureHeightRational
    "HorizontalOffsetAsRational", // kCMFormatDescriptionKey_CleanApertureHorizontalOffsetRational
    "VerticalOffsetAsRational",   // kCMFormatDescriptionKey_CleanApertureVerticalOffsetRational
    "CVFieldCount",               // kCVImageBufferFieldCountKey
    "CVFieldDetail",              // kCVImageBufferFieldDetailKey
    "TemporalTopFirst",           // kCVImageBufferFieldDetailTemporalTopFirst
    "TemporalBottomFirst",        // kCVImageBufferFieldDetailTemporalBottomFirst
    "SpatialFirstLineEarly",      // kCVImageBufferFieldDetailSpatialFirstLineEarly
    "SpatialFirstLineLate",       // kCVImageBufferFieldDetailSpatialFirstLineLate
    "CVPixelAspectRatio",         // kCVImageBufferPixelAspectRatioKey
    "HorizontalSpacing",          // kCVImageBufferPixelAspectRatioHorizontalSpacingKey
    "VerticalSpacing",            // kCVImageBufferPixelAspectRatioVerticalSpacingKey
    "CVImageBufferColorPrimaries", // kCVImageBufferColorPrimariesKey
    "ITU_R_709_2",                // kCVImageBufferColorPrimaries_ITU_R_709_2
    "EBU_3213",                   // kCVImageBufferColorPrimaries_EBU_3213
    "SMPTE_C",                    // kCVImageBufferColorPrimaries_SMPTE_C
    "CVImageBufferTransferFunction", // kCVImageBufferTransferFunctionKey
    "ITU_R_709_2",                // kCVImageBufferTransferFunction_ITU_R_709_2
    "SMPTE_240M_1995",            // kCVImageBufferTransferFunction_SMPTE_240M_1995
    "UseGamma",                   // kCVImageBufferTransferFunction_UseGamma
    "CVImageBufferGammaLevel",    // kCVImageBufferGammaLevelKey
    "CVImageBufferYCbCrMatrix",   // kCVImageBufferYCbCrMatrixKey
    "ITU_R_709_2",                // kCVImageBufferYCbCrMatrix_ITU_R_709_2
    "ITU_R_601_4",                // kCVImageBufferYCbCrMatrix_ITU_R_601_4
    "SMPTE_240M_1995",            // kCVImageBufferYCbCrMatrix_SMPTE_240M_1995
    "FullRangeVideo",             // kCMFormatDescriptionExtension_FullRangeVideo
    "CVImageBufferICCProfile",    // kCMFormatDescriptionExtension_ICCProfile
    "CVBytesPerRow",              // kCMFormatDescriptionExtension_BytesPerRow
    "CVImageBufferChromaLocationTopField", // kCVImageBufferChromaLocationTopFieldKey
    "CVImageBufferChromaLocationBottomField", // kCVImageBufferChromaLocationBottomFieldKey
    "Left",                       // kCVImageBufferChromaLocation_Left
    "Center",                     // kCVImageBufferChromaLocation_Center
    "TopLeft",                    // kCVImageBufferChromaLocation_TopLeft
    "Top",                        // kCVImageBufferChromaLocation_Top
    "BottomLeft",                 // kCVImageBufferChromaLocation_BottomLeft
    "Bottom",                     // kCVImageBufferChromaLocation_Bottom
    "DV 4:2:0",                   // kCVImageBufferChromaLocation_DV420
    "TemporalQuality",            // kCMFormatDescriptionExtension_TemporalQuality
    "SpatialQuality",             // kCMFormatDescriptionExtension_SpatialQuality
    "Version",                    // kCMFormatDescriptionExtension_Version
    "RevisionLevel",              // kCMFormatDescriptionExtension_RevisionLevel
    "Vendor",                     // kCMFormatDescriptionExtension_Vendor
    "CommonEncryptionProtected",  // kFigFormatDescriptionExtension_CommonEncryptionProtected
    "CommonEncryptionTrackEncryptionBox", // kFigFormatDescriptionExtension_CommonEncryptionTrackEncryptionBox
    "CommonEncryptionOriginalFormat", // kFigFormatDescriptionExtension_CommonEncryptionOriginalFormat
    "FieldsPerSample",                // kFigFormatDescriptionExtension_FieldsPerSample
    "avcC",
    "d263",
    "uuid",
    "UUID",
    "sinf",
    "pinf",
    "CryptorSubsampleAuxiliaryData", // kCMSampleAttachmentKey_CryptorSubsampleAuxiliaryData
    "hvcC",
    "MetadataKeyTable",     // kCMFormatDescriptionExtensionKey_MetadataKeyTable
    "MetadataKeyNamespace", // kCMMetadataFormatDescriptionKey_Namespace
    "MetadataKeyValue",     // kCMMetadataFormatDescriptionKey_Value
    "MetadataKeyLocalID",   // kCMMetadataFormatDescriptionKey_LocalID
    "MetadataKeyDataTypeNameSpace", // kCMMetadataFormatDescriptionKey_DataTypeNamespace
    "MetadataKeyDataType",  // kCMMetadataFormatDescriptionKey_DataType
    "MetadataKeyConformingDataTypes", // kCMMetadataFormatDescriptionKey_ConformingDataTypes
    "MetadataPrimaryLocalIDs", // kFigFormatDescriptionExtensionKey_MetadataPrimaryLocalIDs
    "MetadataLocalIDDependencyLists", // kFigFormatDescriptionExtensionKey_MetadataLocalIDDependencyLists
    "BitsPerComponent",               // kFigFormatDescriptionExtension_BitsPerComponent
    "MetadataKeyStructuralDependency", // kCMMetadataFormatDescriptionKey_StructuralDependency
    "StructuralDependencyIsInvalidFlag", // kCMMetadataFormatDescription_StructuralDependencyKey_DependencyIsInvalidFlag
    "MetadataKeySetupData",              // kCMMetadataFormatDescriptionKey_SetupData
    "DisplayFlags",                      // kCMTextFormatDescriptionExtension_DisplayFlags
    "BackgroundColor",                   // kCMTextFormatDescriptionExtension_BackgroundColor
    "Red",                               // kCMTextFormatDescriptionColor_Red
    "Green",                             // kCMTextFormatDescriptionColor_Green
    "Blue",                              // kCMTextFormatDescriptionColor_Blue
    "Alpha",                             // kCMTextFormatDescriptionColor_Alpha
    "DefaultTextBox",                    // kCMTextFormatDescriptionExtension_DefaultTextBox
    "Top",                               // kCMTextFormatDescriptionRect_Top
    "Left",                              // kCMTextFormatDescriptionRect_Left
    "Bottom",                            // kCMTextFormatDescriptionRect_Bottom
    "Right",                             // kCMTextFormatDescriptionRect_Right
    "DefaultStyle",                      // kCMTextFormatDescriptionExtension_DefaultStyle
    "StartChar",                         // kCMTextFormatDescriptionStyle_StartChar
    "Font",                              // kCMTextFormatDescriptionStyle_Font
    "FontFace",                          // kCMTextFormatDescriptionStyle_FontFace
    "ForegroundColor",                   // kCMTextFormatDescriptionStyle_ForegroundColor
    "FontSize",                          // kCMTextFormatDescriptionStyle_FontSize
    "HorizontalJustification", // kCMTextFormatDescriptionExtension_HorizontalJustification
    "VerticalJustification",   // kCMTextFormatDescriptionExtension_VerticalJustification
    "EndChar",                 // kCMTextFormatDescriptionStyle_EndChar
    "FontTable",               // kCMTextFormatDescriptionExtension_FontTable
    "TextJustification",       // kCMTextFormatDescriptionExtension_TextJustification
    "Height",                  // kCMTextFormatDescriptionStyle_Height
    "Ascent",                  // kCMTextFormatDescriptionStyle_Ascent
    "DefaultFontName",         // kCMTextFormatDescriptionExtension_DefaultFontName
    "HorizontalJustification", // kCMTextFormatDescriptionExtension_HorizontalJustification
    "VerticalJustification",   // kCMTextFormatDescriptionExtension_VerticalJustification
    "TimeCode_SourceReferenceName", // kCMTimeCodeFormatDescriptionExtension_SourceReferenceName
    "TimeCode_SourceReferenceName_Value", // kCMTimeCodeFormatDescriptionKey_Value
    "TimeCode_SourceReferenceName_LangCode", // kCMTimeCodeFormatDescriptionKey_LangCode
    "Linear",                  // kCMFormatDescriptionTransferFunction_Linear
    "SMPTE_ST_428_1",          // kCMFormatDescriptionTransferFunction_SMPTE_ST_428_1
    "SMPTE_ST_2084_PQ",        // kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ
    "ITU_R_2100_HLG",          // kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG
    "IEC_sRGB",                // kCMFormatDescriptionTransferFunction_sRGB
    "AlphaChannelIsOpaque",    // kCVImageBufferAlphaChannelIsOpaque
    "CGColorSpace",            // kCVImageBufferCGColorSpaceKey
    "CVImageBufferICCProfile", // kCVImageBufferICCProfileKey
    "CVImageBufferColorPrimaries", // kCVImageBufferColorPrimariesKey
    "CVImageBufferTransferFunction", // kCVImageBufferTransferFunctionKey
    "CVImageBufferYCbCrMatrix", // kCVImageBufferYCbCrMatrixKey
    "ColorInfoGuessedBy",
    "QTMovieTime", // kCVBufferMovieTimeKey
    "TimeValue",   // kCVBufferTimeValueKey
    "TimeScale",   // kCVBufferTimeScaleKey
    "SceneReferredExtendedLinear",
    "SourceTrackID",                   // kFigSampleBufferAttachmentKey_SourceTrackID
    "CameraIntrinsicMatrix",           // kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix
    "DroppedFrameReason",              // kCMSampleBufferAttachmentKey_DroppedFrameReason
    "DroppedFrameReasonInfo",          // kCMSampleBufferAttachmentKey_DroppedFrameReasonInfo
    "StillImageLensStabilizationInfo", // kCMSampleBufferAttachmentKey_StillImageLensStabilizationInfo
    "HEVCSyncSampleNALUnitType",       // kCMSampleAttachmentKey_HEVCSyncSampleNALUnitType
    "AmbientViewingEnvironment",       // kCMFormatDescriptionExtension_AmbientViewingEnvironment
    "auxi",
    "ccst",
    "slmC",
    "IsGradualDecoderRefreshAuthoritative", // kCMSampleBufferAttachmentKey_IsGradualDecoderRefreshAuthoritative
    "EndOfSiriTTSUtterance",                // kCMSampleBufferAttachmentKey_EndOfSiriTTSUtterance
    "FECGroupID",                           // kVTSampleAttachmentKey_FECGroupID
    "FECLastFrameInGroup",                  // kVTSampleAttachmentKey_FECLastFrameInGroup
    "FECLevelOfProtection",                 // kVTSampleAttachmentKey_FECLevelOfProtection
    "QualityMetrics",                       // kVTSampleAttachmentKey_QualityMetrics
    "EncodedFrameAvgQP",                    // kVTSampleAttachmentKey_EncodedFrameAvgQP
    "PadByteCount",                         // kVTSampleAttachmentKey_PadByteCount
    "VRAWidth",
    "VRAHeight",
    "CVImageBufferChromaSubsampling", // kCVImageBufferChromaSubsamplingKey
    "dvvC",
    "disc",
    "TierFlag",                                  // kCMHEVCTemporalLevelInfoKey_TierFlag
    "ProfileCompatibilityFlags", // kCMHEVCTemporalLevelInfoKey_ProfileCompatibilityFlags
    "ProfileSpace",              // kCMHEVCTemporalLevelInfoKey_ProfileSpace
    "TemporalLevel",             // kCMHEVCTemporalLevelInfoKey_TemporalLevel
    "ConstraintIndicatorFlags",  // kCMHEVCTemporalLevelInfoKey_ConstraintIndicatorFlags
    "LevelIndex",                // kCMHEVCTemporalLevelInfoKey_LevelIndex
    "ProfileIndex",              // kCMHEVCTemporalLevelInfoKey_ProfileIndex
    "HEVCTemporalLevelInfo",     // kCMSampleAttachmentKey_HEVCTemporalLevelInfo
    "HEVCTemporalSubLayerAccess", // kCMSampleAttachmentKey_HEVCTemporalSubLayerAccess
    "HEVCStepwiseTemporalSubLayerAccess", // kCMSampleAttachmentKey_HEVCStepwiseTemporalSubLayerAccess
    "HEVCSyncSampleNALUnitType",          // kCMSampleAttachmentKey_HEVCSyncSampleNALUnitType
    "AudioIndependentSampleDecoderRefreshCount", // kCMSampleAttachmentKey_AudioIndependentSampleDecoderRefreshCount
    "av1C",
    "MasteringDisplayColorVolume", // kCVImageBufferMasteringDisplayColorVolumeKey
    "ContentLightLevelInfo",       // kCVImageBufferContentLightLevelInfoKey
    "HDR10PlusPerFrameData",       // kCMSampleAttachmentKey_HDR10PlusPerFrameData
    "vpcC",
];

/// The string at `index`, if the table has one.
pub fn key_at(index: u16) -> Option<&'static str> {
    KNOWN_KEYS.get(usize::from(index)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_key_reads_at_each_index() {
        assert_eq!(key_at(46), Some("Width"));
        assert_eq!(key_at(55), Some("Width"));
        assert_eq!(key_at(47), Some("Height"));
        assert_eq!(key_at(148), Some("Height"));
    }

    #[test]
    fn the_ends_of_the_table_are_where_the_serializer_has_them() {
        assert_eq!(key_at(0), Some("NotSync"));
        assert_eq!(key_at(9), Some("CryptorIV"));
        assert_eq!(key_at(211), Some("vpcC"));
        assert_eq!(key_at(212), None);
    }
}
