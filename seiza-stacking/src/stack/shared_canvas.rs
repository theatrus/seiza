//! A channel's photometric reference on another mosaic's immutable geometry.

use super::*;
use std::collections::HashMap;

/// Lights and their calibration masters for one filter in a joint canvas.
/// All groups must use compatible optics and the same processing options.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CanvasChannel {
    /// Light paths, in planning order. The first group's first light anchors geometry.
    pub paths: Vec<PathBuf>,
    /// Optional master bias belonging to this group.
    pub bias: Option<PathBuf>,
    /// Optional master dark belonging to this group.
    pub dark: Option<PathBuf>,
    /// Optional master flat belonging to this filter.
    pub flat: Option<PathBuf>,
    /// Exposure override for this group's master dark, in seconds.
    pub dark_exposure_seconds: Option<f64>,
}

impl LiveStacker {
    /// Plan the union of all groups' connected footprints, then start an
    /// empty stack for `channel` (zero-based). Each group is calibrated with
    /// its own masters. Geometry planning disables photometric normalization
    /// across filters; the selected channel then builds its own reference
    /// using the requested normalization and weighting. Push its lights
    /// explicitly. Use [`Self::open_fits_on_canvas`] on the result for other
    /// channels, avoiding another joint plan.
    ///
    /// Paths must be unique across groups. Registration failures are omitted
    /// from the union; disconnected fields cannot be joined. Groups must use
    /// compatible optics and processing options. The progress callback covers
    /// both union planning and selected-channel preparation (count restarts).
    pub fn open_fits_joint_canvas(
        channels: &[CanvasChannel],
        channel: usize,
        options: StackOptions,
        maximum_samples: usize,
        mut progress: impl FnMut(&Path, usize, usize, usize) -> crate::Continue,
    ) -> Result<Self> {
        let selected = channels
            .get(channel)
            .ok_or_else(|| Error::Stack("invalid canvas channel index".into()))?;
        let options = Self::expanding_options(options)?;
        let mut paths = Vec::new();
        let mut owners = HashMap::new();
        let mut calibrations = Vec::new();
        for (index, group) in channels.iter().enumerate() {
            if group.paths.is_empty() {
                return Err(Error::Stack(
                    "every canvas channel needs light frames".into(),
                ));
            }
            if group
                .dark_exposure_seconds
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
                || (group.dark_exposure_seconds.is_some() && group.dark.is_none())
            {
                return Err(Error::Stack(
                    "a positive master-dark exposure requires a dark path".into(),
                ));
            }
            for path in &group.paths {
                // Calibration files may be shared between filters, but no
                // light may be a master in any group or occur twice.
                for other in channels {
                    distinct_input_paths([
                        Some(path.as_path()),
                        other.bias.as_deref(),
                        other.dark.as_deref(),
                        other.flat.as_deref(),
                    ])?;
                }
                if owners.insert(path_identity(path), index).is_some() {
                    return Err(Error::Stack(
                        "joint canvas light paths must be unique".into(),
                    ));
                }
                paths.push(path.clone());
            }
            calibrations.push(CalibrationMasters::from_fits_paths(
                group.bias.as_deref(),
                group.dark.as_deref(),
                group.flat.as_deref(),
                group.dark_exposure_seconds,
            )?);
        }
        let mut geometry_options = options.clone();
        geometry_options.normalization = NormalizationMode::None;
        geometry_options.weighting = FrameWeighting::Equal;
        let plan = crate::expand::plan_prepared(
            &paths,
            &|path, frame, bin| {
                prepare_light_at(
                    frame,
                    &calibrations[owners[&path_identity(path)]],
                    &geometry_options.match_tolerances,
                    &geometry_options,
                    bin,
                )
                .map_err(LightFailure::into_error)
            },
            &geometry_options,
            maximum_samples,
            &mut progress,
        )?;
        drop(calibrations);
        let source = Self::from_expansion(plan, CalibrationMasters::default(), geometry_options)?;
        source.open_fits_on_canvas(
            &selected.paths,
            selected.bias.as_deref(),
            selected.dark.as_deref(),
            selected.flat.as_deref(),
            selected.dark_exposure_seconds,
            options,
            progress,
        )
    }

    /// Start an empty channel stack on this stack's expanded canvas. The
    /// canvas dimensions, panel geometry, lens correction and WCS are reused;
    /// no pixels or brightness measurements from this stack are integrated.
    /// `paths` builds a new normalization reference using only this channel's
    /// calibrated lights. Push every light explicitly after this returns.
    ///
    /// Plan the common footprint first with [`Self::open_fits_expanding`]. It
    /// must cover the desired output field: this call never grows that grid,
    /// and samples outside it are cropped. A saved/reopened expanded context
    /// can also supply the canvas. The bin factor must match the canvas's.
    /// The first light anchors this channel's brightness; later lights with
    /// no photometric overlap are retried after other lights fill the gap.
    /// Lights that cannot register or normalize are rejected when pushed.
    /// Read, calibration, and cancellation errors abort construction.
    #[allow(clippy::too_many_arguments)]
    pub fn open_fits_on_canvas<P: AsRef<Path>>(
        &self,
        paths: &[P],
        bias_path: Option<&Path>,
        dark_path: Option<&Path>,
        flat_path: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
        options: StackOptions,
        mut progress: impl FnMut(&Path, usize, usize, usize) -> crate::Continue,
    ) -> Result<Self> {
        let source_canvas = self.canvas_registrar.as_ref().ok_or_else(|| {
            Error::Stack("a shared mosaic canvas requires an expanded stack".into())
        })?;
        let mut options = Self::expanding_options(options)?;
        if options.bin_factor != self.options.bin_factor {
            return Err(Error::Stack(
                "shared canvas bin factor must match its source".into(),
            ));
        }
        if paths.is_empty() {
            return Err(Error::Stack(
                "a shared canvas needs channel light frames".into(),
            ));
        }
        let masters = [bias_path, dark_path, flat_path];
        for path in paths {
            distinct_input_paths(std::iter::once(Some(path.as_ref())).chain(masters))?;
        }
        let calibration = CalibrationMasters::from_fits_paths(
            bias_path,
            dark_path,
            flat_path,
            dark_exposure_seconds,
        )?;
        let (width, height, channels) = (
            self.reference.width,
            self.reference.height,
            self.reference.channels,
        );
        options.registration.maximum_drift_pixels = options
            .scale()
            .native_length((width as f64).hypot(height as f64));
        // Retain measured panel stars and projections, while honoring this
        // channel's registration settings and GPU policy.
        let mut canvas = crate::expand::CanvasRegistrar::new(
            &self.reference,
            &source_canvas.regions(),
            Some(source_canvas.panel_stars()),
            source_canvas.lens(),
            &options.registration,
            options.scale(),
            &options.gpu,
        )?
        .with_projections(source_canvas.panel_projections())?;
        let registrar = canvas.anchor_registrar();
        let mut image = LinearImage::new(
            width,
            height,
            channels,
            vec![f32::NAN; self.reference.sample_count()],
        )?;
        let mut metadata = None;
        let mut headers = Vec::new();
        let mut noise = Vec::new();
        let mut registrations = HashMap::new();
        let stamps: HashMap<_, _> = paths
            .iter()
            .map(|path| {
                let path = path_identity(path.as_ref());
                let stamp = crate::replay::SourceStamp::of(&path);
                (path, stamp)
            })
            .collect();
        let check = |path: &Path| -> Result<()> {
            if stamps[&path_identity(path)] != crate::replay::SourceStamp::of(path) {
                return Err(Error::Stack(format!(
                    "{} changed during shared canvas planning",
                    path.display()
                )));
            }
            Ok(())
        };
        let mut pending: Vec<_> = paths.iter().map(|p| p.as_ref()).collect();
        let mut completed = 0;
        let mut largest_frame = (0, 0);
        while !pending.is_empty() {
            let before = pending.len();
            let mut deferred = Vec::new();
            for path in pending {
                if progress(path, completed, width, height) == crate::Continue::No {
                    return Err(Error::Cancelled);
                }
                check(path)?;
                let frame = FitsFrame::open(path)?;
                let frame_metadata = frame.metadata();
                let (frame, cfa) =
                    prepare_light(frame, &calibration, &options.match_tolerances, &options)
                        .map_err(LightFailure::into_error)?;
                check(path)?;
                let first = metadata.is_none();
                largest_frame.0 = largest_frame.0.max(frame.image.width);
                largest_frame.1 = largest_frame.1.max(frame.image.height);
                let channel_noise = if first && !options.weighting.is_equal() {
                    measure_reference_noise(&frame.image)?
                } else {
                    Vec::new()
                };
                let registration = canvas.register(&frame.image);
                let key = path_identity(path);
                let result = match registration {
                    Ok(registration) => {
                        let geometry_options = StackOptions {
                            normalization: if first {
                                NormalizationMode::None
                            } else {
                                options.normalization
                            },
                            weighting: FrameWeighting::Equal,
                            ..options.clone()
                        };
                        let ready = prepare_frame(
                            if first { &self.reference } else { &image },
                            FrameRegistration {
                                registrar: &registrar,
                                canvas: Some(&canvas),
                                known: Some(Ok(&registration)),
                                ceilings: &std::sync::OnceLock::new(),
                                #[cfg(feature = "gpu")]
                                fit_tiles: &std::sync::OnceLock::new(),
                            },
                            &geometry_options,
                            &[],
                            frame.image,
                            cfa,
                            None,
                            None,
                        )?;
                        match ready {
                            PreparedFrame::Ready(ready) => {
                                if first {
                                    metadata = Some(frame_metadata);
                                    noise = channel_noise;
                                    headers = frame.headers;
                                    headers.retain(|(key, _)| !crate::fits::preserve_wcs_key(key));
                                    headers.extend(
                                        self.reference_headers
                                            .iter()
                                            .filter(|(key, _)| crate::fits::preserve_wcs_key(key))
                                            .cloned(),
                                    );
                                }
                                let region = ready.region;
                                let mut filled = false;
                                for row in 0..region.height {
                                    let start = ((region.y + row) * width + region.x) * channels;
                                    let source = row * region.width * channels;
                                    for (target, value) in image.data
                                        [start..start + region.width * channels]
                                        .iter_mut()
                                        .zip(
                                            &ready.registered.data
                                                [source..source + region.width * channels],
                                        )
                                    {
                                        if !target.is_finite() && value.is_finite() {
                                            *target = *value;
                                            filled = true;
                                        }
                                    }
                                }
                                if filled
                                    && matches!(
                                        options.normalization,
                                        NormalizationMode::LocalBackground { .. }
                                    )
                                {
                                    canvas.update_photometry(&image);
                                }
                                completed += 1;
                                Ok(registration)
                            }
                            PreparedFrame::Rejected(reason) => Err(reason.to_string()),
                        }
                    }
                    Err(error) => Err(error.to_string()),
                };
                if let (true, Err(reason)) = (first, &result) {
                    return Err(Error::Stack(format!(
                        "channel anchor could not join the shared canvas: {reason}"
                    )));
                }
                if result.is_err() {
                    deferred.push(path);
                }
                registrations.insert(
                    key.clone(),
                    crate::expand::PlannedRegistration::shared(result, stamps[&key]),
                );
            }
            if deferred.len() == before {
                break;
            }
            pending = deferred;
        }
        for path in paths {
            check(path.as_ref())?;
        }
        if progress(paths[0].as_ref(), completed, width, height) == crate::Continue::No {
            return Err(Error::Cancelled);
        }
        let rejected = registrations
            .values()
            .filter(|plan| plan.result.is_err())
            .count();
        // Keep omissions from union planning visible on every channel.
        let mut notes = source_canvas.planning_notes().to_vec();
        if rejected != 0 {
            notes.push(format!(
                "{rejected} channel lights could not register or normalize on the shared canvas and will be rejected when pushed"
            ));
        }
        canvas.update_photometry(&image);
        canvas = canvas
            .with_largest_frame(largest_frame)
            .with_summary(source_canvas.lens_summary())
            .with_planning_notes(notes);
        let mut stacker = Self::from_expansion(
            crate::expand::Expansion {
                image,
                headers,
                metadata: metadata.expect("the anchor was admitted"),
                noise,
                registrations,
                canvas,
            },
            calibration,
            options,
        )?;
        stacker.input_paths = distinct_input_paths(masters)?;
        stacker
            .ledger
            .set_current_calibration(crate::replay::CalibrationRecord::from_paths(
                bias_path,
                dark_path,
                flat_path,
                dark_exposure_seconds,
            ));
        Ok(stacker)
    }
}
