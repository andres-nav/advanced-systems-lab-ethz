<script setup>
import { computed } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const props = defineProps({
  eq: {
    type: String,
    required: true
  },
  display: Boolean
})

const renderedMath = computed(() => {
  return katex.renderToString(props.eq, {
    displayMode: props.display,
    throwOnError: false
  })
})
</script>

<template>
  <span v-html="renderedMath" class="math-container"></span>
</template>
