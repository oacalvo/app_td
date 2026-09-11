# Developer Notes

This file is a maintenance map for future code changes. It is intentionally shorter and more direct than the user README, so a developer can route a request to the right files without rereading the whole project.

## Working Environment

Use the existing virtual environment in this workspace:

```bash
source ~/venvs/image_paper_uds/bin/activate
python -m unittest discover -s tests
```

The root-level scripts are launch shims:

- `td_mosaic_app.py` imports and runs `app_code/td_mosaic_app.py`
- `td_wavelet_filter.py` imports the local wavelet filter module
- `inspect_cube_xy_time.py` imports the local cube inspector

Large session files, `manifest.json`, `status.json`, PNG outputs, FITS outputs, and autosaves are data/products. Do not treat them as source code unless the task is explicitly about saved session compatibility or exported output schema.

Runtime/deployment note: this is a local Tk desktop application, not a hosted service. There is currently no Docker, web server, or package/deploy configuration; deployment means copying/cloning the source to a machine with Python `3.10+`, the dependencies from `requirements.txt`, Tk, and access to the user-provided FITS cube.

## Source Map

- `app_code/core.py`: data classes, cube loading, axis order normalization, straight/curved cut geometry, bilinear sampling, TD construction, display helpers.
- `app_code/td_mosaic_app.py`: GUI, state, sessions, cuts, stacks, studies, background jobs, NUWT orchestration, wavelet event review, linking, metrics, exports.
- `app_code/td_wavelet_filter.py`: wavelet ridge detection, mode segmentation, sine fitting, physical parameters, candidate decisions.
- `app_code/nuwt/`: Python port of the Auto-NUWT tracking routines.
- `tests/`: focused regression tests for session flow, study generation, core geometry, wavelet filtering, and wavelet overlays.

## Main Data Flow

1. Load FITS cube and normalize it to `(time, y, x)` in `core.load_cube`.
2. Store user cuts as `Cut` objects.
3. Convert a cut plus panel settings into a TD image with `core.compute_td`.
4. Run NUWT crest tracking on the TD.
5. Run the wavelet filter on NUWT threads.
6. Convert wavelet candidates into editable event records.
7. Review, trim, split, accept/reject, lock, link, and classify events.
8. Export maps, TDs, event tables, trace tables, manual velocity trace tables, study outputs, and reports.

## Change Routing

Use this section first when deciding what to inspect.

### UI Layout And Tool Tabs

Look in `app_code/td_mosaic_app.py`:

- main/top controls: `_build_controls`
- sidebar tabs: `_build_sidebar`
- detached TD window tabs: `_open_td_window`, `_select_td_window_tool_tab`, `_on_td_window_tool_tab_changed`

Current sidebar tabs are `Session / Export`, `Cuts / Geometry`, `TD / Stacks`, `Slope Velocity`, `Waves`, and `Robust Study`. The detached TD window editor tabs are `TD / Cut`, `Slope Velocity`, and `Waves`.

`Session / Export` buttons are intentionally gated in `_refresh_export_controls()`: TD cut exports need cuts and a FITS/PNG mode, wave trace exports need wavelet events, velocity exports need manual velocity traces, and macro tables need both wave/event data and manual velocity data in the selected scope.

Manual velocity tracing has dedicated debug flags:

- `TD_MOSAIC_DEBUG_VELOCITY=1` prints click, record-build, table-refresh, redraw, and finish timings.
- `TD_MOSAIC_DEBUG_VELOCITY_MOTION=1` additionally prints mouse-motion preview redraws and is intentionally noisy.

Keep `velocity_trace_table_updating` around programmatic `Treeview` selection changes. Without it, `selection_set()` can trigger `<<TreeviewSelect>>`, which can call `_refresh_td_window()` and re-enter the velocity table refresh loop.

### Cube Loading And Axis Order

Look in:

- `app_code/core.py`: `normalize_cube_axis_order`, `load_cube`
- `app_code/td_mosaic_app.py`: startup/session cube recovery
- `tests/test_session_flow.py`

### Straight And Curved TD Geometry

Look in:

- `app_code/core.py`: `cut_polyline_points`, `polyline_arc_lengths`, `polyline_point_at_length`, `sampled_cut_geometry`, `cut_point_at_distance`, `compute_td`
- `app_code/td_mosaic_app.py`: `_point_on_cut_distance`, `_cut_geometry_for_frame`, `_cut_preview`
- `tests/test_core_geometry.py`

Rules:

- Straight cuts are sampled along endpoint distance.
- Curved cuts are sampled by accumulated arc length.
- Curved cut width uses local perpendicular offsets from the tangent.
- Dynamic keyframed geometry currently applies only to straight cuts.

### Cut Editing And Drawing

Look in `app_code/td_mosaic_app.py`:

- straight cut creation: `_start_draw_cut`, `_finish_pending_cut`, `_create_cut`
- curved cut creation: `_start_draw_curve_cut`, `_append_curve_cut_point`, `_finish_pending_curve_cut`
- function curve creation: `_start_draw_function_cut`, `_evaluate_parametric_function_cut_expression`
- shared edits: `_apply_cut_points`, `_set_cut_angle`, `_set_cut_length`, `_shift_cut_geometry`
- dynamic cuts: `_set_cut_dynamic_keyframe`, `_cut_geometry_for_frame`, `_dynamic_cut_geometry_samples`

### TD Panels And TD Display

Look in `app_code/td_mosaic_app.py`:

- per-cut TD state: `_cut_td_params`, `_seed_cut_td_params_from_panel`, `_sync_panels_from_cut_td_params`
- TD cache/build: `_panel_td`, `_cut_td`
- rendering: `_draw_td_axis`, `_draw_td_panels`, `_refresh_td_window`
- coordinate conversion: `_td_window_plot_to_td_coords`

### NUWT Tracking

Look in:

- `app_code/td_mosaic_app.py`: `_parse_td_window_crest_tracking_params`, `_run_td_window_crest_tracking`, `_sync_td_window_crest_tracking`
- `app_code/nuwt/`: core tracking implementation

Do not mix NUWT with wavelet review. NUWT produces crest-like tracked threads. Wavelet analysis decides which thread segments are oscillatory events.

### Wavelet Segmentation And Oscillation Cuts

Look in:

- `app_code/td_wavelet_filter.py`: `_ridge_power_segments`, `wavelet_select_segment`, `analyze_tracked_segment_with_wavelet`, `analyze_tracked_threads_with_wavelets`
- `app_code/td_mosaic_app.py`: `_wavelet_worker`, `_replacement_wavelet_run_payload`, `_keep_distinct_wavelet_segments`, `_best_wavelet_segment`
- `tests/test_wavelet_filter.py`
- `tests/test_td_mosaic_wavelet_overlay.py`

Common causes of shortened oscillations:

- `max_jump_pix` splits the source NUWT thread.
- `segment_power_frac` crops to high-power ridge support.
- `min_points_segment` and `min_points_cut_seg` reject short pieces.
- Duplicate removal should only remove strongly overlapping candidates within the same mode. It must not collapse distinct modes from the same source trace.

### Wavelet Event Review

Look in `app_code/td_mosaic_app.py`:

- event creation: `_make_td_window_wavelet_event`
- status/counting: `_td_window_wavelet_event_status`, `_td_window_wavelet_event_is_counted`
- confidence/QA: `_wavelet_event_confidence_details`, `_td_window_wavelet_event_qa_flags`
- edit actions: `_accept_td_window_selected_wavelet_event`, `_reject_td_window_selected_wavelet_event`, `_trim_td_window_selected_wavelet_event`, `_split_td_window_selected_wavelet_event`
- table refresh: `_refresh_td_window_wavelet_table`, `_td_window_wavelet_event_row`

### Manual Velocity Traces

Look in `app_code/td_mosaic_app.py`:

- `_build_velocity_trace_record`
- `_build_velocity_trace_record_from_points`
- `_velocity_trace_mode_key`
- `_finish_td_window_velocity_trace`
- `_undo_td_window_velocity_trace_point`
- `_toggle_td_window_velocity_trace_mode`
- `_refresh_td_window_velocity_trace_table`
- `_draw_td_window_velocity_traces`
- `_on_td_window_press`

These are manual TD measurements independent from NUWT and wavelet fitting. They support straight two-point traces, polyline traces, and quadratic fits with at least three distinct time positions. Physical scale comes from `cad_s`, `res_arcsec_px`, and `km_per_arcsec`; saved traces also carry `km_per_pixel`, `km_s_per_px_frame`, and `km_s2_per_px_frame2`. Check the saved payload fields around `trace_kind`, `points`, `point_count`, segment speed ranges, physical scale, and quadratic acceleration before changing exports or session compatibility.

### Event Trace Tables And Dynamics

Look in `app_code/td_mosaic_app.py`:

- `_event_trace_rows`
- `_thread_trace_rows`
- `_write_trace_event_fits`
- `_export_trace_ids_fits`
- `_experiment_curated_rows`

Trace rows map TD coordinates back to frame index, distance in pixels, map coordinates, and current cut endpoints. Use these rows for lifetime, displacement, and map-position history.

### Stacks, Link Groups, And Propagation

Look in `app_code/td_mosaic_app.py`:

- stacks: `_create_stack_state`, `_selected_stack_cut_ids`, `_open_stack_browser`
- event linking: `_link_wavelet_events_by_cut_refs`, `_collect_wavelet_link_groups`
- propagation class: `_set_wavelet_event_propagation_class`, `_set_stack_browser_event_class`
- exports: `_export_propagation_tables`

Stacks are only groups of cuts. Link groups are the manual claim that several events are the same physical wave.

### Robust / Exhaustive Studies

The UI labels this workflow as `Robust Study` / `Robust Cube Study`; older function names and tests still use `experiment` or `exhaustive`.

Look in `app_code/td_mosaic_app.py`:

- state defaults: `_make_default_experiment_state`
- parameter extraction: `_experiment_study_td_params`, `_experiment_study_crest_params`, `_experiment_study_wavelet_params`
- generation: `_generate_exhaustive_experiment`
- workers: `_run_selected_experiment`, `_experiment_worker`, `_experiment_tables_worker`
- output finalization: `_finalize_experiment_outputs`, `_write_experiment_stack_payloads`
- tests: `tests/test_exhaustive_study.py`

Rules:

- Study base cuts must be straight.
- Standard studies shift the base cut and rotate each member.
- Full-cube angular studies generate a family of parallel cuts for each angle.
- Study outputs are intentionally file-heavy and include master/cell/trace/velocity/novelty/important-zone tables. Avoid changing export schemas without checking rebuild/export tests or adding new tests.

### Exports And Reports

Look in `app_code/td_mosaic_app.py`:

- FITS/PNG cut exports: `_write_cut_td_fits`, `_write_cut_quicklook_png`, `_export_cut_ids_fits`
- trace exports: `_write_trace_event_fits`, `_export_trace_ids_fits`
- curated tables: `_export_curated_results`
- reports: `_export_curated_report`
- metrics: `_export_metrics_table`, `_export_metrics_figure`
- study tables: `_export_selected_experiment_tables`

## Testing Guide

Run all tests after code changes:

```bash
source ~/venvs/image_paper_uds/bin/activate
python -m unittest discover -s tests
```

For narrow changes, start with the closest test file:

- geometry/cuts: `python -m unittest tests.test_core_geometry`
- wavelet internals: `python -m unittest tests.test_wavelet_filter`
- wavelet GUI/event payloads: `python -m unittest tests.test_td_mosaic_wavelet_overlay`
- studies: `python -m unittest tests.test_exhaustive_study`
- sessions/startup: `python -m unittest tests.test_session_flow`

Then run the full suite before finishing.

## Maintenance Rules

- Prefer adding focused tests before changing event filtering, geometry, or study output behavior.
- Keep generated data files out of code review unless the task is specifically about saved outputs.
- Do not collapse wavelet candidates across different `mode_rank` values.
- Do not assume panel state and cut state are separate; many panel operations sync back to per-cut analysis state.
- Keep README UI names aligned with `_build_sidebar` and detached TD tab labels when moving controls.
- Do not add dynamic geometry support for curved cuts without updating `core.compute_td`, trace mapping, tests, and README limitations together.
- If a request mentions lifetime, displacement, or velocity, first decide whether it means manual velocity traces, NUWT source traces, wavelet event traces, or study-level trace tables.
