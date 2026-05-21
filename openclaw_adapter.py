"""OpenClaw compatibility adapter for STMClaw.

This module provides a bridge for OpenClaw to launch STMClaw and handle
basic lifecycle events such as graceful shutdown and session cleanup.
"""

import logging
import os
import signal
import threading

from Auto_scan import main as _auto_scan_main

SHUTDOWN_EVENT = threading.Event()


def _get_navigation_instruction():
    return os.getenv(
        'STMCLAW_NAVIGATION_INSTRUCTION',
        'The scan shall start from the top-left and proceed in a serpentine pattern until the entire area is covered.',
    )


def _signal_handler(signum, frame):
    logging.info('OpenClaw adapter received termination signal %s', signum)
    SHUTDOWN_EVENT.set()


def _setup_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _safe_shutdown(engine):
    if engine is None:
        return

    try:
        logging.info('Requesting STMClaw stop and withdraw.')
        if hasattr(engine, 'StopScanAndWithdraw'):
            engine.StopScanAndWithdraw()
    except Exception:
        logging.exception('Error while requesting scan stop.')

    try:
        if hasattr(engine, 'save_checkpoint'):
            engine.save_checkpoint()
    except Exception:
        logging.exception('Error while saving checkpoint during shutdown.')

    try:
        if hasattr(engine, 'close'):
            engine.close()
    except Exception:
        logging.exception('Error while closing STMClaw engine.')


def run_stmclaw():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    logging.info('Starting STMClaw OpenClaw adapter')
    _setup_signal_handlers()

    engine = None
    had_error = False

    try:
        engine = _auto_scan_main(
            shutdown_event=SHUTDOWN_EVENT,
            navigation_instruction=_get_navigation_instruction(),
        )
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt received; shutting down STMClaw.')
        had_error = True
    except SystemExit as exc:
        logging.info('SystemExit received: %s', exc)
        had_error = True
    except Exception:
        logging.exception('Unhandled exception in STMClaw runtime.')
        had_error = True
        raise
    finally:
        if SHUTDOWN_EVENT.is_set() or had_error:
            _safe_shutdown(engine)
        logging.info('STMClaw OpenClaw adapter exiting')

    return engine


if __name__ == '__main__':
    run_stmclaw()
