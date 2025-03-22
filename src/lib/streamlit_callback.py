import inspect
from typing import Any, Callable, Dict, Optional, TypeVar

import streamlit
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for Streamlit."""

    def __init__(self, container: DeltaGenerator):
        """Initialize callback handler."""
        self.container = container

        # Auxiliar variable
        self.states_messages = []

        # Inicializar los placeholders inmediatamente
        self.current_status = container.container()
        self.status_container = None
        self.text_placeholder = container.empty()
        self.node_name = None

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        if "context" not in inputs and len(self.states_messages) == 0:
            status_message = "Buscando información relevante"
            self.states_messages.append(status_message)
            self.status_container = self.current_status.status(
                status_message, state="running"
            )
        elif (
            "context" in inputs
            and "response" not in inputs
            and len(self.states_messages) == 1
        ):
            status_message = "Generando respuesta"
            self.states_messages.append(status_message)
            self.status_container.update(state="complete")
            self.status_container = self.current_status.status(
                status_message, state="running"
            )
        elif (
            "context" in inputs
            and "response" in inputs
            and len(self.states_messages) == 2
        ):
            status_message = "Validando respuesta"
            self.states_messages.append(status_message)
            self.status_container.update(state="complete")
            self.current_status.status(status_message, state="running")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.text_placeholder is not None:
            self.text_placeholder.markdown(token, unsafe_allow_html=True)

    def on_custom_event(
        self, name, data, *, run_id, tags=None, metadata=None, **kwargs
    ):
        print("custom: ", name, data)
        pass


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that integrates fully with any LangChain ChatLLM integration,
    updating the provided Streamlit container with outputs such as tokens, model responses,
    and intermediate steps. This function ensures that all callback methods run within
    the Streamlit execution context, fixing the NoSessionContext() error commonly encountered
    in Streamlit callbacks.

    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered
                                           during the LLM interaction.
    Returns:
        BaseCallbackHandler: An instance of StreamlitCallbackHandler configured for full integration
                             with ChatLLM, enabling dynamic updates in the Streamlit app.
    """

    # Define a type variable for generic type hinting in the decorator, ensuring the original
    # function and wrapped function maintain the same return type.
    fn_return_type = TypeVar("fn_return_type")

    # Decorator function to add Streamlit's execution context to a function
    def add_streamlit_context(
        fn: Callable[..., fn_return_type],
    ) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        This is necessary for interacting with Streamlit components from within callback functions
        and prevents the NoSessionContext() error by adding the correct session context.

        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated, typically a callback method.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        # Retrieve the current Streamlit script execution context.
        # This context holds session information necessary for Streamlit operations.
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            If the Streamlit context is not set, it can lead to NoSessionContext() errors, which this
            wrapper resolves by ensuring that the correct context is used when the function runs.

            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            # Add the previously captured Streamlit context to the current execution.
            # This step fixes NoSessionContext() errors by ensuring that Streamlit knows which session
            # is executing the code, allowing it to properly manage session state and updates.
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of Streamlit's StreamlitCallbackHandler with the provided Streamlit container
    st_cb = StreamlitCallbackHandler(parent_container)

    # Iterate over all methods of the StreamlitCallbackHandler instance
    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith(
            "on_"
        ):  # Identify callback methods that respond to LLM events
            # Wrap each callback method with the Streamlit context setup to prevent session errors
            setattr(
                st_cb, method_name, add_streamlit_context(method_func)
            )  # Replace the method with the wrapped version

    # Return the fully configured StreamlitCallbackHandler instance, now context-aware and integrated with any ChatLLM
    return st_cb
