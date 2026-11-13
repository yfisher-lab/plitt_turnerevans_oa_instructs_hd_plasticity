# Methods additions and revisions

Draft text for *Octopamine instructs head direction plasticity* (Plitt, Turner-Evans, Co, Layden, Eddison,
Ray, Jayaraman & Fisher; bioRxiv 2025.12.11.693783v1). Two new subsections, and replacement text for two
existing ones that describe how the bump is centered.

Notation follows the paper's existing *Data processing for two-photon imaging* subsection: $Y$ is the
(16 × *T*) matrix of ROI fluorescence over time and $\vec{y}_t$ its column at time *t*.

Passages in **[brackets]** need a value or a figure number filled in, or a decision confirmed; they are
listed again at the end of this file.

---

## New subsection: Bump width

> Suggested placement: in *Data processing for two-photon imaging*, immediately after
> *Population vector analyses*, since it uses the same preprocessed $Y$ and introduces the
> phase-nulling operation that the revised centering subsections below refer back to.

To measure the angular width of the activity bump independently of its position, we first removed the
bump's position from each time point. The spatial Fourier transform of $\vec{y}_t$ was computed along the
ROI axis, and every spatial harmonic *k* was multiplied by $e^{-ik\phi_t}$, where $\phi_t$ is the phase of
the first harmonic. The first harmonic of a 16-ROI ring profile is the population vector, so $\phi_t$ is
the PVA angle and this operation circularly shifts the profile by that angle, to sub-ROI precision,
without the interpolation error that rounding to the nearest ROI would introduce. The inverse transform
was then rolled by 8 ROIs so that the bump sits at the center of the profile. Time points at which no bump
could be measured were excluded: these are time points where the ratio of the first-harmonic magnitude to
the across-ROI mean fluorescence was at or below 0.05, which in practice removed fewer than 0.5% of time
points. The remaining phase-nulled profiles were averaged over time, giving one mean bump profile per
imaging session.

Bump width was measured from this mean profile as a full width at half maximum (FWHM), with no
parametric fit. The 16-point profile was circularly interpolated to 256 points by zero-padding its Fourier
spectrum (splitting the Nyquist component evenly between positive and negative frequencies), the
interpolated profile was rolled so that its peak lay at the first sample, and the half-maximum level was
taken to be halfway between the profile's minimum and its peak. Moving outwards from the peak in each
direction, the first sample below the half-maximum level was found and the crossing located by linear
interpolation between it and the preceding sample. The FWHM is the distance between the two crossings,
converted to radians at $2\pi/256$ radians per sample; a profile that never falls below its half maximum
is undefined and was excluded.

Within a fly, widths from repeated imaging sessions of the same condition were combined before the
conditions were compared. **[Confirm: for the within-genotype comparison of darkness and closed loop
(Fig. X) the session profiles of each fly and condition were averaged and a single width measured from
that mean profile, whereas for the comparisons between genotypes (Fig. Y) a width was measured from each
session and the widths were averaged. Either is defensible, but the manuscript should state one procedure
or make the difference explicit.]** Only flies contributing to both conditions were included. Darkness was
compared with the **[second]** closed-loop block, so that the two conditions come from imaging sessions
adjacent in the experiment. Widths were compared between conditions within a fly with a Wilcoxon
signed-rank test, and between genotypes with Mann-Whitney U tests of the closed-loop width and of the
within-fly change in width from darkness to closed loop, Holm-corrected within each family of tests.

---

## New subsection: *Ex vivo* GFlamp2 imaging

> Suggested placement: a subsection of its own after *Response to optogenetic stimulation*. The text below
> covers only the processing of the imaging data; the dissection, bath and explant conditions are not
> described here.

**[Dissection, saline composition, explant mounting and bath temperature to be supplied — these are not
recoverable from the analysis code.]**

GFlamp2 fluorescence was recorded from the **[ER / EPG neuropil]** while EL neurons expressing CsChrimson
were stimulated optogenetically. Imaging used the same volumetric z-scanning as the *in vivo* experiments
(6 z-planes), at a volume rate of **[7.7]** Hz. Optogenetic stimulation was delivered through the
microscope's mark-points module as a single 10 μm spiral centered on the **[ellipsoid body]**, using a
617 nm laser at **[50]%** power. Each session comprised 20 stimulations of 50 ms, beginning 10 s into the
recording and repeating every 10 s. A single region of interest covering the **[ER / EPG neuropil]** was
drawn, and its mean fluorescence in the GFlamp2 channel was extracted from each imaging volume.

Unlike the *in vivo* recordings, these time series were not detrended, background-regressed, smoothed or
z-scored, because each stimulation is quantified relative to its own immediately preceding baseline.
For each stimulation, fluorescence from 2 s before to 8 s after the stimulus was linearly interpolated
onto a common time base sampled every 100 ms, so that trials acquired at slightly different frame times
could be averaged together. Imaging frames acquired while the stimulation laser was on were contaminated
by its light and were discarded (−0.2 s to +0.22 s relative to the stimulus). The baseline $F_0$ for each
stimulation was the mean fluorescence over the samples preceding this excluded window, and the response
was expressed as $\Delta F/F = F/F_0 - 1$. Each stimulation was summarized by the peak of $\Delta F/F$
after the excluded window and by the area under the response curve over the same period (the sum of
$\Delta F/F$ multiplied by the 100 ms sampling interval). The 20 stimulations of a session were averaged
to give one response per session, and one session was analyzed per fly.

We recorded from 11 flies expressing CsChrimson in EL neurons and GFlamp2 in ER neurons, 9 flies
expressing CsChrimson in EL neurons and GFlamp2 in EPG neurons, and 5 flies of each cell type in which
CsChrimson was omitted as a control. Peak $\Delta F/F$ was compared between the experimental and control
preparations of each cell type, and between ER and EPG experimental preparations, using Mann-Whitney U
tests with Holm correction across the three comparisons.

---

## Revised subsection: ER octopamine/dopamine bump visualization (Fig 1n, Extended Data Fig 5)

> Replaces the existing subsection in full. Changes: the heading alignment is no longer binned, the
> across-fly alignment uses the bump phase rather than the index of its peak, and time points are
> restricted to those with appreciable rotation. **[Confirm the rotation-speed criterion — it is in the
> analysis code but not in the current manuscript text.]**

Consider the matrix of ROI fluorescence values over time, $Y$, whose column at time *t* is $\vec{y}_t$. To
average activity in a heading-referenced frame, each $\vec{y}_t$ was circularly shifted so that the fly's
current heading fell on a fixed ROI. Rather than binning the heading into the 16 ROI bins, the heading was
converted to a fractional ROI index and the profile was shifted by that amount using the Fourier shift
theorem, that is, by multiplying each spatial harmonic by a phase ramp and inverting the transform. This
shifts to sub-ROI precision and is exact for band-limited profiles; for a shift by a whole number of ROIs
it reduces to the index shift used previously. We took the temporal average of the heading-aligned
activity for the first and second half of the imaging session separately, using only time points at which
the fly's rotational speed exceeded **[0.5]** rad/s.

To align the resulting profiles across flies we used the bump's phase rather than the index of its peak.
The phase of the first spatial Fourier harmonic of the first half's average, which is its PVA angle, was
removed by the same phase-ramp operation, placing that bump at the center ROI with sub-ROI precision, and
the identical shift was applied to the second half's average. Because both halves are shifted by the same
amount and neither is quantized to whole ROIs, any drift of the bump between the two halves of the session
is preserved, and the alignment itself contributes no rounding of up to half an ROI.

---

## Revised subsection: Difference in EL and EPG PVA angles (Fig 2, Extended Data Fig 9)

> Only the second paragraph changes (the visualization of EL leading EPG during rapid turns). The first
> and third paragraphs are unchanged and are reproduced here for context.

To quantify the average angular distance between the EL and EPG bumps we calculated the temporal average
of the absolute angular difference between the EL PVA angle and EPG PVA angle (Fig. 2i, Extended Data
Fig. 9f). For Extended Data Fig. 9c, we binned the absolute angular difference by the average PVA
magnitude of the EL and EPG populations at each timepoint. Averages were calculated first within flies and
then across flies.

To visualize the extent to which the EL PVA can lead the EPG PVA during rapid turns (Extended Data
Fig. 9a,g), the columns of $Y_{EL}$ and $Y_{EPG}$ were circularly shifted by the phase of the EPG bump.
Each spatial harmonic of both channels was multiplied by $e^{-ik\phi_{t,EPG}}$, where $\phi_{t,EPG}$ is
the phase of the first spatial Fourier harmonic of $\vec{y}_{t,EPG}$, that is, the EPG PVA angle, and the
inverse transforms were rolled by 8 ROIs so that the EPG bump sits at the center of the profile. Shifting
by the phase rather than by $\mathrm{argmax}(\vec{y}_{t,EPG})$ aligns the EPG bump to sub-ROI precision,
so the displacement being measured is not quantized to the $2\pi/16$ radian ROI spacing; applying the same
shift to both channels leaves the angular displacement between the EL and EPG bumps unchanged. Heading
velocity for this analysis was smoothed with a Gaussian kernel of width **[300]** msec. We separately
took the average of timepoints where the fly's rotational velocity was between $\pi$ and $2\pi$ rad/s or
between $-2\pi$ and $-\pi$ rad/s, and min-max normalized the resulting mean profile of each channel.

To quantify the extent to which EL PVA angle leads the EPG PVA angle for all turning speeds (Extended Data
Fig. 9b,h), we calculated the mean PVA angle difference (EL minus EPG) for time points within each
rotational velocity bin. Averages were calculated within each fly and then combined data across flies.

---

## Items to confirm before submission

1. **Bump width, session averaging.** The within-genotype and between-genotype analyses combine sessions
   in different orders (average the profiles, then measure the width; versus measure each session's width,
   then average). One procedure should be adopted, or the difference stated.
2. **Bump width, closed-loop block.** The block compared with darkness differs between experiments
   (the second block in some, the first in others). State which, per figure.
3. **Rotation-speed criterion in the ER bump visualization.** The analysis code restricts the heading-
   aligned average to time points with rotational speed above 0.5 rad/s. This is not in the current
   manuscript text; confirm it was used for the published figure.
4. **Upper bound on turning speed.** The published text says "greater than π rad/s or less than −π rad/s";
   the analysis code also imposes an upper bound of 2π rad/s. The revised text above states the bound.
5. ***Ex vivo* acquisition parameters.** Volume rate, stimulation power and the identity of the stimulated
   point are read from one session's metadata and should be confirmed across sessions; the dissection and
   bath conditions must be written from the experimental record.
6. ***Ex vivo* ROI.** The analysis uses a single ROI, stored as the "outer ring" region of the
   preprocessing pipeline, from the second imaging channel. Describe how that ROI was drawn.
7. **Bump-magnitude threshold.** The exclusion criterion for the bump-width analysis is a ratio of
   first-harmonic magnitude to across-ROI mean fluorescence, which is *not* the PVA magnitude defined in
   *Population vector analyses*. Since it excludes under 0.5% of time points either way, consider dropping
   it from the analysis rather than defining a second quantity in the Methods.
