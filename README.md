# Time-Distance Mosaic

Interactive GUI for exploring a 3D FITS cube, drawing time-distance (TD) cuts, tracking crest-like motions with a Python port of Auto-NUWT, isolating oscillatory segments with wavelets, reviewing detections manually, linking the same wave across cuts, and exporting tables, FITS products, and figures.

## What the app is for

This tool is designed for workflows like:

- load a 3D cube `(time, y, x)` or any supported axis order
- draw one or many cuts across the map
- build TD diagrams from those cuts
- run pure NUWT tracking on each TD
- isolate oscillatory segments with wavelet analysis
- manually accept, reject, trim, split, lock, and annotate candidate events
- link the same physical wave across different cuts
- classify linked groups as `same-wave`, `propagating`, `additional`, or `local`
- compare amplitudes, periods, energies, and related quantities
- export review products, figures, and machine-readable tables

The GUI entry point is [`td_mosaic_app.py`](./td_mosaic_app.py), which calls [`app_code/td_mosaic_app.py`](./app_code/td_mosaic_app.py).

## ☀️ Quick Start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Launch the GUI and choose a FITS cube from the startup dialog:

```bash
python td_mosaic_app.py
```

Launch the GUI with an explicit cube and axis order:

```bash
python td_mosaic_app.py --cube /path/to/cube.fits --cube-order TYX
```

Reopen a saved session and let the app recover the cube path stored in that JSON:

```bash
python td_mosaic_app.py --session /path/to/session.json
```

Your FITS cube is user-provided and is not expected to live inside this repository.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

- `numpy`
- `scipy`
- `astropy`
- `matplotlib`
- `PyWavelets`

The GUI uses `tkinter`, which is part of the Python standard library on many installations and is therefore not listed in `requirements.txt`.

## 🌤️ Launching the app

If `--cube` is omitted, the app opens a startup dialog and lets you choose the FITS cube interactively.

Supported `--cube-order` values:

- `TYX`, `TXY`, `YTX`, `YXT`, `XTY`, `XYT`
- numeric aliases: `123`, `132`, `213`, `231`, `312`, `321`

The app needs a working graphical display. If Tk cannot open, the program exits with a GUI/display error.

## 🌻 Core concepts

### Cube

The input data cube. Internally the app works with a normalized `(t, y, x)` representation.

### Cut

A line segment sampled through the map. Each cut produces a TD diagram over a chosen time range, width, and weighting.

### Feature Axis

A line or curve used to auto-generate many cuts at regular spacing. This is useful when you want systematic angular or spatial sampling.

### Panel

One TD view in the main mosaic. Each panel can point to a different cut and different temporal settings.

### Detached TD Window

A full editor for one panel/cut. This is where the detailed NUWT, wavelet, review, and plotting workflow happens.

### Stack

A named group of cuts. Stacks are useful for browsing related cuts together, especially when scanning angle or position.

### Link Group

A manual grouping of events that you consider to be the same physical wave seen in multiple cuts.

### Propagation Group

A summary view built from curated observations and link groups. It helps compare members, representative events, class labels, and summary statistics.

## Recommended workflow

1. Open the cube.
2. Draw one cut or generate many cuts from a feature axis.
3. Assign cuts to TD panels.
4. Adjust TD settings: time range, stride, width, weighting.
5. Open a detached TD window for the panel of interest.
6. Run `Crest Tracking (NUWT)` first.
7. Run `Wavelet Filter` second.
8. Review the event table manually.
9. Link corresponding events across cuts.
10. Use `Stacks`, `Link Groups`, `Propagation`, and `Wave Statistics` to compare results.
11. Export tables, reports, FITS files, and figures.
12. Save the session.

## UI overview

The left sidebar is organized into these tabs:

- `TD`
- `Cuts`
- `Geometry`
- `Measure`
- `Stacks`
- `Export`

The top action row includes:

- `Save Session`
- `Load Session`
- `Export Curated`
- `Metrics`
- `Export Report`
- `Link Groups`
- `Propagation`
- `Batch Pipeline`
- `Saved FITS`
- `Open Cube`

## Sidebar tabs

### TD

Controls per-panel TD settings such as:

- assigned cut
- `t_ini`, `t_fin`
- `stride`
- `width`
- `weighting`

This is the fast way to build and compare TD panels in the main mosaic.

### Cuts

Main cut management tools:

- `Add Cut`
- `Draw/Replace Cut`
- `Delete Cut`
- `Copy Cut`
- `Paste Cut`
- quick rotations
- `Open Cut Browser`

This tab also contains `Feature Axis / Auto Cuts`, where you can:

- draw a line or curve axis
- choose cut spacing
- choose generated cut length
- set an angle offset relative to the local perpendicular
- optionally create a stack automatically
- generate many cuts at once

This is the main tool for systematic sampling across position or angle.

### Geometry

Direct geometric editing of the selected cut:

- anchor mode
- length mode
- angle
- length
- endpoint coordinates

It also includes `Dynamic Cut`, which allows time-varying geometry using keyframes. This is useful when the feature you want to cut through drifts over time.

### Measure

This tab is for reference/target comparisons and relative control between cuts. Use it when you need to compare one cut against another in a controlled way.

### Stacks

Stacks let you group multiple cuts and browse them as one logical set. This is useful when:

- several cuts sample the same region with different angles
- several cuts scan neighboring spatial positions
- you want to inspect whether the same event persists or changes across the set

### Export

This tab controls:

- export folder
- `Write FITS`
- `Write PNG`
- `Separate folders`

It also exposes:

- `Save current map`
- `Selected cut`
- `Stack`
- `All cuts`
- `Selected cut traces`
- `Stack traces`
- `All traces`
- `Open saved FITS browser`

## Detached TD window

The detached TD window is the main analysis workspace for a single panel/cut.

It contains:

- TD controls
- ROI controls
- explicit cut center and vertices
- direct angle/length editing
- `Crest Tracking (NUWT)`
- `Wavelet Filter`
- `Wavelet Events`

### Pure NUWT section

`Crest Tracking (NUWT)` is intentionally limited to the real tracking controls:

- `cad [s]`
- `res [arcsec/px]`
- `grad`
- `min thread`
- `max dist jump`
- `max time skip`
- `invert`
- `gauss fit (slow)`

This section is meant to stay close to the original Auto-NUWT behavior.

The current Python port was aligned so that crest following behaves like the original Auto-NUWT logic: it follows the first valid crest in search order instead of using an extra ranking heuristic.

### Wavelet Filter section

`Wavelet Filter` is the editable cleanup and isolation stage. It is where you tune how tracked threads are converted into oscillatory events.

Important parameters include:

- period range: `P min`, `P max`
- significance/selection: `power ratio`, `segment frac`
- minimum support: `min pts seg`, `min pts cut`
- amplitude and smoothness filters
- physical conversion: `km / arcsec`
- optional physics: `density [kg/m3]`, `phase speed [km/s]`

It also contains the extra cleanup knobs that are intentionally not part of pure NUWT:

- `min SNR`
- `min prominence`
- `continuity w`
- `time w`
- `quality w`
- `error w`

Use this section when you want the app to be more editable, more selective, or more aggressive at isolating overlapping oscillatory behavior.

### Multi-mode wavelet isolation

The wavelet stage can now keep more than one spectral mode per tracked thread. This matters when:

- two or more oscillations overlap in the same trajectory
- one thread contains multiple characteristic periods
- you want to split a complex signal into separate candidate events

Each candidate stores a `mode` index, and the event table shows it explicitly.

### Running the analysis

Typical order:

1. `Run tracking`
2. `Run wavelet filter`
3. inspect events in the table
4. accept/reject/edit candidates

There is also a `Batch Pipeline` action from the main window that runs NUWT plus wavelet analysis across all assigned panels.

## Wavelet event review

The `Wavelet Events` table is the manual curation layer.

Each row represents one candidate event and includes fields such as:

- status
- origin
- thread id
- segment id
- wavelet segment id
- mode
- period
- frequency
- amplitude
- velocity
- acceleration
- specific energy
- duration
- power ratio
- confidence score
- fit/interpolated point counts
- lock state
- link count
- QA flags
- reason

### Manual review actions

Per-event actions include:

- `Recompute selected`
- `Accept`
- `Reject`
- `Reset`
- `Undo`
- `Redo`
- `Lock/Unlock`
- `Note`
- `History`
- `Trim selected`
- `Split selected`

Bulk review actions include:

- `Accept visible`
- `Reject visible`
- `Reset visible`

This is the section to use when you want to increase the final curated table quality by manual intervention.

### Advanced filtering

The event table can be filtered by:

- review state: `all`, `accepted`, `rejected`, `manual`, `split`
- QA state
- locked/unlocked
- linked/unlinked
- minimum score
- period range
- amplitude range
- energy range

This is useful for cleaning large result sets before exporting or linking.

## Linking the same wave across cuts

The app distinguishes between three different ideas:

### Stack

A stack is only a grouping of cuts. It does not automatically mean the same physical wave.

### Link Group

A link group is your manual statement that several events across cuts correspond to the same wave.

The `Linked Wavelet Groups` window lists:

- groups
- event counts
- cut counts
- counted/locked totals
- mean confidence
- member statuses

And for each member:

- panel
- cut
- event id
- status
- confidence
- lock
- QA flags
- notes

### Propagation Group

The `Propagation Groups` window summarizes curated observations by group and class. It reports things like:

- group id
- class
- member count
- cut count
- representative event
- best-confidence event
- maximum amplitude
- median period
- mean confidence

The default class logic is:

- `local` for a single-member group
- `same-wave` for an unlabeled multi-member group

You can also label events manually as:

- `same-wave`
- `propagating`
- `additional`
- `local`

## Stack browser

The stack browser is a convenient comparison tool for a set of cuts.

It provides:

- one-cut-at-a-time browsing through the stack
- event table for the current cut
- trace-point table
- quick class buttons: `Same`, `Propagating`, `Additional`, `Local`
- `Thread->Stack`
- direct access to `Scatter stats` and `Bar stats`
- direct opening of the full TD editor for the current cut

This is often the best place to decide whether the same wave appears consistently as angle or position changes.

## Statistics and comparisons

The `Wave Statistics` window supports:

- `histogram`
- `scatter`
- `bar`

Available metrics include:

- period
- amplitude in arcsec
- amplitude in km
- velocity amplitude
- acceleration amplitude
- specific energy
- energy flux
- power ratio
- confidence score
- duration
- link count

Grouping modes for bar plots include:

- cut
- status
- propagation class
- panel
- link group

Bar plots use robust error bars based on the 16th and 84th percentiles.

### PNG export

The statistics window includes:

- `Export Figure`
- `Export Master Table`

This supports saving plot outputs such as PNG.

`Scatter stats` and `Bar stats` are available both:

- from the `Stack Browser`
- from the detached TD window

## Reports and tables

### Curated results

`Export Curated` writes the curated event table in machine-readable form.

### Curated report

`Export Report` writes a compact summary report with figure panels and histograms of the curated counted events.

### Master table

The statistics window can export a master table for the current scope. This is the closest thing to the unified observation table of curated events.

### Propagation tables

The propagation window can export:

- observation rows
- grouped propagation summaries

This is the best export if you want a table of linked waves and their grouped properties.

## Saved FITS browser

The `Saved FITS` browser lets you inspect products previously exported by the app from the current export folder. It can preview saved maps, TD products, and table-like outputs.

## Sessions and autosave

The app supports:

- `Save Session`
- `Load Session`
- automatic session autosave

Autosave is written as a JSON session file associated with the current cube. This is useful when the GUI is being used as a long curation environment rather than as a one-shot script.

Session files store both:

- `cube_path`
- `cube_axis_order`

When a session is loaded, the app can reopen the matching FITS cube automatically. If the file moved, the app asks you to locate it.

## Practical interpretation

The intended analysis logic is:

- use `NUWT` to detect and follow crest-like ridges in the TD
- use `Wavelet Filter` to decide which tracked segments are oscillatory and physically relevant
- use manual review to accept/reject/split/trim ambiguous cases
- use `Link Groups` when you decide multiple detections are the same wave
- use `Propagation` to summarize those linked detections
- use `Wave Statistics` and exports to compare amplitudes, periods, energies, and confidence across cuts, stacks, and groups

## Current limitations

- `Propagation` is a summary and classification tool, not a fully automatic physical propagation solver.
- The in-app statistics tools compare well by cut, stack, panel, status, class, and link group.
- If you want dedicated plots directly against geometric quantities such as cut angle or map position `(x, y)`, the current workflow is best done from exported tables or by extending the metrics layer further.
- Manual review is still the final authority for ambiguous or overlapping events.

## Code structure

- [`td_mosaic_app.py`](./td_mosaic_app.py): top-level launcher
- [`app_code/td_mosaic_app.py`](./app_code/td_mosaic_app.py): main GUI and workflow logic
- [`app_code/td_wavelet_filter.py`](./app_code/td_wavelet_filter.py): wavelet segmentation and event extraction
- [`app_code/nuwt`](./app_code/nuwt): Python port of the core Auto-NUWT routines

## Short version

Use `NUWT` for pure crest tracking. Use `Wavelet Filter` for editable isolation, cleanup, and physical characterization. Use the event table for manual curation. Use `Link Groups` and `Propagation` to say which detections are the same wave across cuts. Use `Wave Statistics` plus exported tables to compare how amplitudes, periods, and energies change across your dataset.
