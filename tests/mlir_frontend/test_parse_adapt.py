"""
Adapter tests: same assertions as test_dialects_parse.py, but driven through
MLIRFrontendParser instead of the regex parser.

Each ``TestXxxAdapt`` class inherits the corresponding ``TestXxxParsers`` base
and overrides only tests that rely on regex-parser-specific syntax not accepted
by MLIR (overridden to ``pytest.skip``).  Attribute normalisation (e.g.
arith.cmpi integer predicate → string) is handled by MLIRTypeAdapter handlers,
so the inherited assertions pass unchanged.
"""

import pytest

from test_dialects_parse import (
    TestArithParsers as _TestArithParsers,
    TestLinalgParsers as _TestLinalgParsers,
    TestTensorParsers as _TestTensorParsers,
    TestKtdpParsers as _TestKtdpParsers,
    TestScfParsers as _TestScfParsers,
    TestMathParsers as _TestMathParsers,
)

from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser  # noqa: E402

# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class MLIRFrontendParseTestMixin:
    """Override _parse to drive tests through MLIRFrontendParser."""

    def assert_operand_names(self, op, *names):
        pass  # bindings parser uses positional %argN names — not portable

    def assert_attribute(self, op, key, value, transform=None):
        if key in ("iter_var", "iter_args"):
            # Bindings parser assigns positional names; e.g. for:
            #   func.func @_test(%lb: index, %ub: index, %step: index) {
            #     scf.for %i = %lb to %ub step %step { ... }
            # key="iter_var", op.attributes={"iter_var": "%i"}        (regex)
            # key="iter_var", op.attributes={"iter_var": "%arg3"}     (bindings, %arg0-2 are func args)
            assert key in op.attributes
        else:
            super().assert_attribute(op, key, value, transform=transform)

    def _parse(self, op_text, parse_ctx=None, args=None, prelude=None):
        """Parse ``op_text`` through the MLIR frontend and return the op under test.

        The op is wrapped in a synthetic ``func.func`` so it can reference
        SSA values (``args``) with explicit types. ``prelude`` is optional
        extra op text placed *above* ``op_text`` in the function body — use
        it when the op under test consumes a value whose type cannot appear
        in the function signature (e.g. ``!ktdp.tile_future<T, groups = ...>``,
        whose type parameter contains commas the arg-list parser cannot
        split).

        Wrapper module shape::

            module {
              func.func @_test(<func_args>) attributes { grid = [1] } {
                <prelude>          # only if prelude is not None
                <op_text>          # the op under test — this is what we return
                return
              }
            }

        The ``grid = [1]`` on the wrapper func is inert scaffolding, not part
        of what these tests cover. ``MLIRFrontendParser._build_ir_function``
        reads ``grid`` off the func to populate ``IRFunction.grid``, but the
        MLIR verifier treats it as an opaque ``ArrayAttr`` — it is never
        cross-checked against any op inside the body (e.g. the ``groups``
        affine set on ``!ktdp.tile_future``). Bumping the value here would
        not exercise any additional parse path, and no assertion in this
        file inspects it. End-to-end validation of ``grid > 1`` intertile
        behavior lives in the execution tests (see the ``ring_reduce`` and
        ``ring_reduce_multi_group`` entries in ``tests/conftest.py``), which
        build a ``GridExecutor`` with ``num_cores = math.prod(meta.grid)``
        and check numerical output across cores.

        Returns the parsed ``op_text`` op. It is always the *last* non-return
        op in the body: without a prelude that's the only body op; with a
        prelude the prelude ops come first and ``op_text`` is appended after.

        See ``ParseTestMixin._parse`` for the full ``prelude`` / ``args``
        contract shared with the regex-parser tests.
        """
        # `args` declares external SSA values (name → MLIR type). Validate
        # names against both prelude and op_text — a prelude op may consume
        # a declared value that never appears in op_text itself.
        args = self._resolve_args(f"{prelude or ''}\n{op_text}", args)
        func_args = ", ".join(f"{name}: {mlir_type}" for name, mlir_type in args.items())
        body = f"{prelude}\n    {op_text}" if prelude else op_text
        module_text = f"""\
module {{
  func.func @_test({func_args}) attributes {{ grid = [1] }} {{
    {body}
    return
  }}
}}
"""
        ir_module = MLIRFrontendParser().parse_module(module_text)
        op_under_test = None
        for op in ir_module.get_function("_test").operations:
            if op.op_type not in ("func.return", "return"):
                op_under_test = op
        if op_under_test is None:
            raise RuntimeError(f"No op parsed from:\n{module_text}")
        return op_under_test


# ---------------------------------------------------------------------------
# Arith
# ---------------------------------------------------------------------------

class TestArithAdapt(MLIRFrontendParseTestMixin, _TestArithParsers):
    """Arith tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Linalg
# ---------------------------------------------------------------------------

class TestLinalgAdapt(MLIRFrontendParseTestMixin, _TestLinalgParsers):
    """Linalg tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class TestTensorAdapt(MLIRFrontendParseTestMixin, _TestTensorParsers):
    """Tensor tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Ktdp
# ---------------------------------------------------------------------------

class TestKtdpAdapt(MLIRFrontendParseTestMixin, _TestKtdpParsers):
    """Ktdp tests via MLIRFrontendParser."""

    # test_construct_access_tile: inherited
    # test_construct_access_tile_non_index_elem_type_rejected: inherited
    # test_construct_access_tile_malformed_type_rejected: inherited

    # test_affine_set_with_symbolic_dim: inherited
    # test_construct_memory_view_dynamic_memref_type: inherited
    # test_construct_memory_view_ssa_size_as_operand: inherited

    # test_construct_memory_view_multi_dim_mixed_static_dynamic: inherited

    # test_inter_tile_produce: inherited (groups embedded in tile_future type)
    # test_inter_tile_reduce:  inherited (groups embedded in tile_future type)


# ---------------------------------------------------------------------------
# Scf
# ---------------------------------------------------------------------------


class TestScfAdapt(MLIRFrontendParseTestMixin, _TestScfParsers):
    """Scf tests via MLIRFrontendParser."""

    def test_if_then_else(self):
        # scf.if is supported by the MLIR frontend (the regex parser does not
        # parse it, so this test is frontend-only rather than in the shared
        # base class). operand[0] is the condition; then/else are regions
        # [0]/[1]; no execution-relevant attributes.
        op = self._parse(
            "%r = scf.if %c -> (i32) {\n"
            "      %a = arith.constant 1 : i32\n"
            "      scf.yield %a : i32\n"
            "    } else {\n"
            "      %b = arith.constant 2 : i32\n"
            "      scf.yield %b : i32\n"
            "    }",
            args={"%c": "i1"},
        )
        self.assert_op_type(op, "scf.if")
        self.assert_num_operands(op, 1)
        assert len(op.regions) == 2


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

class TestMathAdapt(MLIRFrontendParseTestMixin, _TestMathParsers):
    """Math tests via MLIRFrontendParser."""
