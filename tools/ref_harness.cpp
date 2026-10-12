// Reference harness for numerical alignment, living entirely in our repository.
//
// It compiles against the read-only reference checkout's *internal* headers and
// links its already-built static library, so it can do two things the published
// Go driver cannot:
//
//   transition  call motionbricks::detail::run_transition directly, which
//               returns the decoded motion *before* restore_world() and
//               apply_context_blend() - the only way to tell a decode bug apart
//               from a post-processing bug.
//   agent       run the full agent plan with the playback seam switchable.
//   encode      dump motionbricks::detail::encode_context for a given context.
//
// Nothing in the reference repository is modified; only its headers and
// prebuilt archives are consumed.
//
// Usage:
//   ref_harness transition CONSTRAINTS.bin OUT.json
//   ref_harness agent BUNDLE STYLE ROOTS.bin ROTATIONS.bin FRAMES MX MY MZ \
//                     FX FY FZ SEED ARGMAX SKIP_BLEND OUT.json
//   ref_harness encode BUNDLE ROOTS.bin ROTATIONS.bin FRAMES OUT.json

#include "agent.hpp"
#include "handles.hpp"
#include "motion_rep.hpp"
#include "planner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml.h>
#endif

namespace {

using motionbricks::detail::transition_constraints;

std::vector<float> read_floats(const char * path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char *>(values.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(stream.gcount()) != count * sizeof(float)) {
        std::fprintf(stderr, "%s: short read\n", path);
        std::exit(1);
    }
    return values;
}

void write_array(std::ostream & out, const char * name, const float * values, std::size_t count) {
    out << '"' << name << "\":[";
    for (std::size_t index = 0; index < count; ++index) {
        if (index) out << ',';
        out << values[index];
    }
    out << ']';
}

void write_motion(std::ostream & out, const mb_motion & motion) {
    out.precision(9);
    out << "{\"frames\":" << motion.frames << ",\"joints\":" << motion.joints << ',';
    write_array(out, "roots", motion.root_translations.data(), motion.root_translations.size());
    out << ',';
    write_array(out, "rotations", motion.local_rotations_xyzw.data(),
                motion.local_rotations_xyzw.size());
    out << ',';
    write_array(out, "target_roots", motion.target_root_translations.data(),
                motion.target_root_translations.size());
    out << ',';
    write_array(out, "target_rotations", motion.target_local_rotations_xyzw.data(),
                motion.target_local_rotations_xyzw.size());
    out << "}\n";
}

int run_transition_mode(const char * constraints_path, const char * out_path) {
    // Layout: global(8*5) local(8*4) poses(8*303) has_global(8) has_local(8)
    // has_poses(8) allowed(11) seed(1) argmax(1) target_roots(4*3)
    // target_rotations(4*34*4)
    constexpr std::size_t global_count = 8U * 5U;
    constexpr std::size_t local_count = 8U * 4U;
    constexpr std::size_t pose_count = 8U * 303U;
    const std::size_t total = global_count + local_count + pose_count + 8U + 8U + 8U + 11U + 2U
        + 4U * 3U + 4U * 34U * 4U;
    std::vector<float> values = read_floats(constraints_path, total);
    std::size_t offset = 0;
    transition_constraints constraints;
    std::memcpy(constraints.global_root.data(), values.data() + offset, global_count * sizeof(float));
    offset += global_count;
    std::memcpy(constraints.local_root.data(), values.data() + offset, local_count * sizeof(float));
    offset += local_count;
    std::memcpy(constraints.poses.data(), values.data() + offset, pose_count * sizeof(float));
    offset += pose_count;
    for (std::size_t index = 0; index < 8U; ++index)
        constraints.has_global_root[index] = values[offset + index] != 0.0F ? 1U : 0U;
    offset += 8U;
    for (std::size_t index = 0; index < 8U; ++index)
        constraints.has_local_root[index] = values[offset + index] != 0.0F ? 1U : 0U;
    offset += 8U;
    for (std::size_t index = 0; index < 8U; ++index)
        constraints.has_poses[index] = values[offset + index] != 0.0F ? 1U : 0U;
    offset += 8U;
    for (std::size_t index = 0; index < 11U; ++index)
        constraints.allowed_tokens[index] = values[offset + index] != 0.0F ? 1U : 0U;
    offset += 11U;
    const auto seed = static_cast<std::uint64_t>(values[offset]);
    const bool argmax = values[offset + 1] != 0.0F;
    offset += 2U;
    std::memcpy(constraints.target_root_translations.data(), values.data() + offset,
                4U * 3U * sizeof(float));
    offset += 4U * 3U;
    std::memcpy(constraints.target_local_rotations_xyzw.data(), values.data() + offset,
                4U * 34U * 4U * sizeof(float));

    std::string reason;
    mb_motion motion;
    motionbricks::detail::transition_trace trace;
    // The model is required for its normalization tables, codebook and
    // skeleton, but the transition itself only needs the runtime inside it.
    mb_model * raw_model = nullptr;
    if (!std::getenv("REF_HARNESS_BUNDLE")) {
        std::fprintf(stderr, "set REF_HARNESS_BUNDLE to the g1-f32 directory\n");
        return 2;
    }
    char error[1024] = {0};
    if (mb_model_load(std::getenv("REF_HARNESS_BUNDLE"), nullptr, &raw_model, error, sizeof(error))
            != MB_OK) {
        std::fprintf(stderr, "model load: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_model, void (*)(mb_model *)> model(raw_model, mb_model_free);

    const auto status = motionbricks::detail::run_transition(
        *model, constraints, motion, nullptr, &trace, reason, seed, argmax);
    if (status != MB_OK) {
        std::fprintf(stderr, "run_transition: %s\n", reason.c_str());
        return 1;
    }
    std::ofstream out(out_path);
    write_motion(out, motion);
    out.flush();
    out.close();
    {
        // Append the auxilliary diagnostics the golden test needs (the same
        // values write_transition_debug records for the agent path).
        std::string text = [&] {
            std::ifstream reader(out_path);
            std::stringstream buffer;
            buffer << reader.rdbuf();
            return buffer.str();
        }();
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        std::ofstream rewritten(out_path);
        rewritten.precision(9);
        rewritten << text.substr(0, text.size() - 1) << ',';
        std::vector<float> tokens(trace.pose_tokens.begin(), trace.pose_tokens.end());
        std::vector<float> logits = trace.duration_logits;
        write_array(rewritten, "pose_tokens", tokens.data(), tokens.size());
        rewritten << ',';
        write_array(rewritten, "duration_logits", logits.data(), logits.size());
        rewritten << "}\n";
    }
    std::printf("wrote %s: frames=%llu\n", out_path,
                static_cast<unsigned long long>(motion.frames));
    return 0;
}

int run_encode_mode(int argc, char ** argv) {
    // encode_context is declared in src/motion_rep.hpp, so the reference's own
    // encoder output can be dumped for a given agent context.
    if (argc != 7) {
        std::fprintf(stderr, "encode mode needs 4 arguments\n");
        return 2;
    }
    const char * bundle = argv[2];
    const char * roots_path = argv[3];
    const char * rotations_path = argv[4];
    const auto frames = static_cast<std::uint64_t>(std::strtoull(argv[5], nullptr, 10));
    const char * out_path = argv[6];
    char error[1024] = {0};
    mb_model * raw_model = nullptr;
    if (mb_model_load(bundle, nullptr, &raw_model, error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "model load: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_model, void (*)(mb_model *)> model(raw_model, mb_model_free);
    std::vector<float> roots = read_floats(roots_path, frames * 3U);
    std::vector<float> rotations = read_floats(rotations_path, frames * 34U * 4U);
    motionbricks::detail::encoded_frames encoded;
    std::string reason;
    const auto status = motionbricks::detail::encode_context(
        *model, roots, rotations, frames, encoded, reason);
    if (status != MB_OK) {
        std::fprintf(stderr, "encode_context: %s\n", reason.c_str());
        return 1;
    }
    std::ofstream out(out_path);
    out.precision(9);
    out << '{';
    write_array(out, "global_root", encoded.global_root.data(), encoded.global_root.size());
    out << ',';
    write_array(out, "local_root", encoded.local_root.data(), encoded.local_root.size());
    out << ',';
    write_array(out, "poses", encoded.poses.data(), encoded.poses.size());
    out << "}\n";
    std::printf("wrote %s\n", out_path);
    return 0;
}

int run_bench_mode(int argc, char ** argv) {
    // Load once, then time N identical agent plans (the same work the MLX
    // benchmark measures) so the two can be compared honestly.
    if (argc != 9) {
        std::fprintf(stderr, "bench mode needs 6 arguments\n");
        return 2;
    }
    const char * bundle = argv[2];
    const char * style_path = argv[3];
    const char * roots_path = argv[4];
    const char * rotations_path = argv[5];
    const auto frames = static_cast<std::uint64_t>(std::strtoull(argv[6], nullptr, 10));
    const int steps = std::atoi(argv[7]);
    const auto seed = static_cast<std::uint64_t>(std::strtoull(argv[8], nullptr, 10));

    char error[1024] = {0};
    mb_model * raw_model = nullptr;
    if (mb_model_load(bundle, nullptr, &raw_model, error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "model load: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_model, void (*)(mb_model *)> model(raw_model, mb_model_free);
    mb_style * raw_style = nullptr;
    if (mb_style_load(model.get(), style_path, &raw_style, error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "style load: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_style, void (*)(mb_style *)> style(raw_style, mb_style_free);
    mb_agent * raw_agent = nullptr;
    if (mb_agent_create(model.get(), &raw_agent, error, sizeof(error)) != MB_OK) return 1;
    std::unique_ptr<mb_agent, void (*)(mb_agent *)> agent(raw_agent, mb_agent_free);
    std::vector<float> roots = read_floats(roots_path, frames * 3U);
    std::vector<float> rotations = read_floats(rotations_path, frames * 34U * 4U);
    if (mb_agent_set_context(agent.get(), roots.data(), rotations.data(), frames, 34U,
                             error, sizeof(error)) != MB_OK) return 1;

    std::vector<double> timings;
    for (int index = 0; index < steps; ++index) {
        mb_command * raw_command = nullptr;
        if (mb_command_create(&raw_command, error, sizeof(error)) != MB_OK) return 1;
        std::unique_ptr<mb_command, void (*)(mb_command *)> command(raw_command, mb_command_free);
        command->style = style.get();
        command->seed = seed + static_cast<std::uint64_t>(index);
        mb_motion * raw_motion = nullptr;
        const auto started = std::chrono::steady_clock::now();
        const auto call_status = mb_agent_plan(agent.get(), command.get(), &raw_motion,
                                               error, sizeof(error));
        const auto finished = std::chrono::steady_clock::now();
        if (call_status != MB_OK) {
            std::fprintf(stderr, "plan: %s\n", error);
            return 1;
        }
        std::unique_ptr<mb_motion, void (*)(mb_motion *)> motion(raw_motion, mb_motion_free);
        timings.push_back(std::chrono::duration<double, std::milli>(finished - started).count());
    }
    std::sort(timings.begin(), timings.end());
    const auto at = [&](double fraction) {
        const auto index = static_cast<std::size_t>(fraction * (timings.size() - 1) + 0.5);
        return timings[std::min(index, timings.size() - 1)];
    };
    std::printf("device ggml-cpu steps %d\n", steps);
    std::printf("  per-plan ms   p50 %.1f   p90 %.1f   max %.1f   min %.1f\n",
                at(0.5), at(0.9), timings.back(), timings.front());
    return 0;
}

int run_agent_mode(int argc, char ** argv) {
    if (argc != 17) {
        std::fprintf(stderr, "agent mode needs 15 arguments\n");
        return 2;
    }
    const char * bundle = argv[2];
    const char * style_path = argv[3];
    const char * roots_path = argv[4];
    const char * rotations_path = argv[5];
    const auto frames = static_cast<std::uint64_t>(std::strtoull(argv[6], nullptr, 10));
    const float movement[3] = {std::strtof(argv[7], nullptr), std::strtof(argv[8], nullptr),
                               std::strtof(argv[9], nullptr)};
    const float facing[3] = {std::strtof(argv[10], nullptr), std::strtof(argv[11], nullptr),
                             std::strtof(argv[12], nullptr)};
    const auto seed = static_cast<std::uint64_t>(std::strtoull(argv[13], nullptr, 10));
    const auto argmax = static_cast<std::uint32_t>(std::strtoul(argv[14], nullptr, 10));
    const auto skip_blend = static_cast<std::uint32_t>(std::strtoul(argv[15], nullptr, 10));
    const char * out_path = argv[16];

    char error[1024] = {0};
    mb_model * raw_model = nullptr;
    if (mb_model_load(bundle, nullptr, &raw_model, error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "model load: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_model, void (*)(mb_model *)> model(raw_model, mb_model_free);
    mb_style * raw_style = nullptr;
    if (mb_style_load(model.get(), style_path, &raw_style, error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "style load: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_style, void (*)(mb_style *)> style(raw_style, mb_style_free);
    mb_agent * raw_agent = nullptr;
    if (mb_agent_create(model.get(), &raw_agent, error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "agent create: %s\n", error);
        return 1;
    }
    std::unique_ptr<mb_agent, void (*)(mb_agent *)> agent(raw_agent, mb_agent_free);

    std::vector<float> roots = read_floats(roots_path, frames * 3U);
    std::vector<float> rotations = read_floats(rotations_path, frames * 34U * 4U);
    if (mb_agent_set_context(agent.get(), roots.data(), rotations.data(), frames, 34U,
                             error, sizeof(error)) != MB_OK) {
        std::fprintf(stderr, "set context: %s\n", error);
        return 1;
    }
    mb_command * raw_command = nullptr;
    if (mb_command_create(&raw_command, error, sizeof(error)) != MB_OK) return 1;
    std::unique_ptr<mb_command, void (*)(mb_command *)> command(raw_command, mb_command_free);
    command->style = style.get();
    command->movement_direction = {movement[0], movement[1], movement[2]};
    command->facing_direction = {facing[0], facing[1], facing[2]};
    command->seed = seed;
    command->sampling_argmax = argmax;
    command->skip_context_blend = skip_blend;

    mb_motion motion;
    std::string reason;
    const auto status = motionbricks::detail::plan_agent(*agent, *command, motion, reason);
    if (status != MB_OK) {
        std::fprintf(stderr, "plan_agent: %s\n", reason.c_str());
        return 1;
    }
    std::ofstream out(out_path);
    write_motion(out, motion);
    std::printf("wrote %s: frames=%llu\n", out_path,
                static_cast<unsigned long long>(motion.frames));
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s transition|agent ...\n", argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "transition") == 0) {
        if (argc != 4) { std::fprintf(stderr, "transition CONSTRAINTS.bin OUT.json\n"); return 2; }
        return run_transition_mode(argv[2], argv[3]);
    }
    if (std::strcmp(argv[1], "agent") == 0) return run_agent_mode(argc, argv);
    if (std::strcmp(argv[1], "encode") == 0) return run_encode_mode(argc, argv);
    if (std::strcmp(argv[1], "bench") == 0) return run_bench_mode(argc, argv);
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
