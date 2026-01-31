"""
Tests for style presets and prompts.
"""

from __future__ import annotations

import pytest

from blackwell_flow.styles import (
    STYLE_PRESETS,
    StylePreset,
    cycle_style,
    get_refinement_prompt,
    get_style_preset,
)


class TestStylePresets:
    """Tests for style presets."""

    def test_all_styles_defined(self) -> None:
        """Test that all expected styles are defined."""
        expected_styles = ["professional", "casual", "creative", "bullet_points"]

        for style in expected_styles:
            assert style in STYLE_PRESETS
            assert isinstance(STYLE_PRESETS[style], StylePreset)

    def test_professional_preset(self) -> None:
        """Test professional style preset."""
        preset = STYLE_PRESETS["professional"]

        assert preset.name == "Professional"
        assert "formal" in preset.description.lower()
        assert "filler words" in preset.system_prompt.lower()
        assert "ONLY the corrected text" in preset.system_prompt

    def test_casual_preset(self) -> None:
        """Test casual style preset."""
        preset = STYLE_PRESETS["casual"]

        assert preset.name == "Casual"
        assert "casual" in preset.description.lower() or "informal" in preset.description.lower()

    def test_creative_preset(self) -> None:
        """Test creative style preset."""
        preset = STYLE_PRESETS["creative"]

        assert preset.name == "Creative"
        assert "creative" in preset.description.lower() or "expressive" in preset.description.lower()

    def test_bullet_points_preset(self) -> None:
        """Test bullet points style preset."""
        preset = STYLE_PRESETS["bullet_points"]

        assert preset.name == "Bullet Points"
        assert "bullet" in preset.system_prompt.lower()


class TestGetStylePreset:
    """Tests for get_style_preset function."""

    def test_get_valid_style(self) -> None:
        """Test getting a valid style preset."""
        preset = get_style_preset("professional")

        assert isinstance(preset, StylePreset)
        assert preset.name == "Professional"

    def test_get_all_styles(self) -> None:
        """Test getting all valid styles."""
        styles = ["professional", "casual", "creative", "bullet_points"]

        for style in styles:
            preset = get_style_preset(style)
            assert isinstance(preset, StylePreset)

    def test_get_invalid_style(self) -> None:
        """Test getting an invalid style raises KeyError."""
        with pytest.raises(KeyError):
            get_style_preset("invalid_style")  # type: ignore


class TestGetRefinementPrompt:
    """Tests for get_refinement_prompt function."""

    def test_prompt_includes_transcript(self) -> None:
        """Test that the prompt includes the transcript."""
        transcript = "This is a test transcript with um some filler words."
        prompt = get_refinement_prompt("professional", transcript)

        assert transcript in prompt

    def test_prompt_includes_system_prompt(self) -> None:
        """Test that the prompt includes the system prompt."""
        transcript = "Test transcript."
        prompt = get_refinement_prompt("professional", transcript)

        preset = STYLE_PRESETS["professional"]
        assert preset.system_prompt in prompt

    def test_prompt_format(self) -> None:
        """Test the prompt has correct format."""
        transcript = "Test transcript."
        prompt = get_refinement_prompt("casual", transcript)

        assert "Transcript:" in prompt
        assert "Corrected text:" in prompt


class TestCycleStyle:
    """Tests for cycle_style function."""

    def test_cycle_from_professional(self) -> None:
        """Test cycling from professional."""
        next_style = cycle_style("professional")
        assert next_style == "casual"

    def test_cycle_from_casual(self) -> None:
        """Test cycling from casual."""
        next_style = cycle_style("casual")
        assert next_style == "creative"

    def test_cycle_from_creative(self) -> None:
        """Test cycling from creative."""
        next_style = cycle_style("creative")
        assert next_style == "bullet_points"

    def test_cycle_from_bullet_points(self) -> None:
        """Test cycling from bullet_points wraps to professional."""
        next_style = cycle_style("bullet_points")
        assert next_style == "professional"

    def test_full_cycle(self) -> None:
        """Test cycling through all styles."""
        style = "professional"
        seen_styles = [style]

        for _ in range(4):
            style = cycle_style(style)
            if style not in seen_styles:
                seen_styles.append(style)

        # After 4 cycles, we should be back to professional
        assert style == "professional"
        assert len(seen_styles) == 4
