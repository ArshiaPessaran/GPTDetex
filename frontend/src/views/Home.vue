<template>
  <div class="home">
    <h3 class="text-light">Enter text here</h3>
    <Main/>
  </div>
</template>

<script>
// @ is an alias to /src
import Main from '@/components/Main.vue'

export default {
  name: 'Home',
  components: {
    Main
  }
}
</script>
