# BharatRAG dashboard UI audit

## Scope

This audit covers visual presentation only. Evaluation APIs, data flows, tab
structure, and user actions remain unchanged.

## Issues found

| Area | Previous issue | Resolution |
| --- | --- | --- |
| Design language | The dark, multi-colour hero conflicted with the neutral application canvas and sidebar. | Replaced it with a restrained surface panel and an indigo hierarchy marker. |
| Palette | CSS, status chips, and Plotly each introduced off-palette values such as slate, coral, purple, and different teal/red/amber shades. | Centralised semantic visual tokens in `dashboard/theme.py` and reused them in CSS, cards, and charts. |
| Sidebar | A universal white-text rule also affected nested controls, making select styling brittle. | Scoped text, captions, controls, and radio states independently. |
| Forms | Native Streamlit controls were visually disconnected from cards and had no clear focus treatment. | Applied white surfaces, border tokens, readable placeholders, and a 3px indigo focus ring. |
| Buttons | Only radius and weight were styled, leaving primary/secondary states and disabled behaviour inconsistent. | Added indigo primary, outline secondary, hover motion, shadow, disabled, and keyboard-focus states. |
| Layout | The hero, tabs, cards, and main container used unrelated spacing values. | Added a consistent spacing rhythm, max content width, responsive container padding, and tab treatment. |
| Status feedback | Good, moderate, and poor used values outside the requested palette. | Uses green, amber, and red semantic tokens with matching low-emphasis surfaces. |
| Analytics | Chart colours did not align with application feedback semantics. | Correct/success series use green, hallucinated/risk series use red, and neutral evaluation charts use indigo. |

## Colour system

| Token | Value | Use |
| --- | --- | --- |
| Primary | `#4F46E5` | Primary actions, active states, neutral evaluation data |
| Primary hover | `#4338CA` | Primary-action hover |
| Accent | `#14B8A6` | Secondary analytical emphasis |
| Success | `#10B981` | Good scores and correct-answer comparisons |
| Warning | `#F59E0B` | Moderate scores |
| Danger | `#EF4444` | Poor scores and hallucinated-answer comparisons |
| Background | `#F8FAFC` | Application canvas |
| Surface | `#FFFFFF` | Cards, forms, tables, expanders |
| Sidebar | `#0F172A` | Product navigation |
| Primary text | `#0F172A` | Headings and controls |
| Secondary text | `#64748B` | Supporting copy and labels |
| Borders | `#E2E8F0` | Control, card, and table separation |

## Accessibility decisions

- Primary body text on white and slate sidebar text on the dark navigation use
  high-contrast foreground/background pairs.
- Placeholder text is supplementary, not the only field label; visible Streamlit
  labels remain present for every input.
- Keyboard users receive an explicit indigo focus outline on buttons, inputs,
  textareas, and tabs.
- Hover effects are additive: controls retain their borders and labels without
  hover, and no workflow depends on colour alone.
- Status chips include text labels (`Good`, `Moderate`, `Poor`) in addition to
  their semantic colour.

## Component changes

- **Sidebar:** grouped labels, native control surfaces, and radio hover states
  create a clearer navigation hierarchy.
- **Hero:** reduced visual weight lets the evaluation workspace—not decoration—
  lead the page.
- **Forms and data editor:** white, bordered controls visually align input with
  results and make focus/cursor position apparent.
- **Buttons:** one clear primary action per flow and quiet outline navigation
  controls reduce accidental emphasis.
- **Metric cards:** a shared white surface, 12px radius, compact progress line,
  and semantic status chip increase scanability.
- **Tabs, tables, expanders, and charts:** aligned border, spacing, and text
  tokens create continuity across native and custom UI.
